# ↑ %%writefile app.py はGitHubには含めないでください！
import streamlit as st
# --- Set page config FIRST ---
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")

# --- Libraries ---
from openai import OpenAI
import vertexai # Vertex AI をインポート
from vertexai.preview.vision_models import ImageGenerationModel # Imagenモデルをインポート
import requests # 画像ダウンロードに必要
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import time
import base64
import os

# --- Function to load prompts from files ---
def load_prompt(file_path):
    """指定されたファイルパスからプロンプトを読み込む"""
    if not os.path.exists(file_path):
        st.error(f"エラー: プロンプトファイルが見つかりません: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"プロンプトファイル読み込みエラー ({file_path}): {e}")
        return None

# --- Load Prompts from Files ---
layout_analysis_prompt_text = load_prompt("prompts/layout_analysis_prompt.txt")
dalle_instruction_template_text = load_prompt("prompts/dalle_prompt_instruction_template.txt") # 名前はそのまま流用
prompts_loaded = layout_analysis_prompt_text is not None and dalle_instruction_template_text is not None

# --- Load API Keys & GCP Config ---
secrets_ok = False
GCP_PROJECT_ID = None
GCP_REGION = None
OPENAI_API_KEY = None

if prompts_loaded:
    try:
        GCP_PROJECT_ID = st.secrets["GCP_PROJECT_ID"]
        GCP_REGION = st.secrets["GCP_REGION"]
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # GPT-4o用に必要
        # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Vertex AIでは通常不要だが存在確認はしておく
        if "GOOGLE_API_KEY" not in st.secrets:
             st.warning("Secretsに GOOGLE_API_KEY がありません（現在は未使用）")


        if not GCP_PROJECT_ID or not GCP_REGION or not OPENAI_API_KEY:
            st.error("エラー: Secretsにキー(GCP_PROJECT_ID, GCP_REGION, OPENAI_API_KEY)が不足しています。")
            st.stop()
        secrets_ok = True
    except KeyError as e:
        st.error(f"エラー: Secretsにキー '{e}' が見つかりません。")
        st.stop()
    except Exception as e:
         st.error(f"Secrets読み込み中に予期せぬエラー: {e}")
         st.stop()

# --- API Function Definitions ---
if secrets_ok and prompts_loaded:

    # analyze_layout_with_gpt4o (変更なし - OpenAIを使用)
    def analyze_layout_with_gpt4o(image_bytes, api_key, layout_prompt_text):
        if not layout_prompt_text: st.error("レイアウト解析プロンプト未読込"); return None
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try: img = Image.open(BytesIO(image_bytes)); img_format = img.format
            except: img_format = None
            if img_format == 'PNG': mime_type = "image/png"
            elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
            else: mime_type = "image/jpeg"
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": layout_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}]}]
            response = client.chat.completions.create(model="gpt-4o", messages=prompt_messages, max_tokens=1000)
            if response.choices and response.choices[0].message.content: return response.choices[0].message.content.strip()
            else: st.warning(f"GPT-4o応答なし(レイアウト解析). R: {response}"); return None
        except Exception as e: st.error(f"GPT-4o APIエラー(レイアウト解析): {e}"); return None


    # generate_dalle_prompt_with_gpt4o (変更なし - OpenAIを使用)
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text):
        """OpenAI GPT-4o APIを呼び出し、画像生成用プロンプトを生成する関数"""
        if not dalle_instruction_template_text: st.error("画像生成指示テンプレート未読込"); return None
        try:
            client = OpenAI(api_key=api_key)
            prompt_generation_instruction = dalle_instruction_template_text.format(
                 size=size, layout_info=layout_info, impression=impression, details=details
            )
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt_generation_instruction}], max_tokens=1500
            )
            if response.choices and response.choices[0].message.content: return response.choices[0].message.content.strip()
            else: st.warning(f"GPT-4o応答なし(プロンプト生成). R: {response}"); return None
        except Exception as e: st.error(f"GPT-4o APIエラー(プロンプト生成): {e}"); return None


    # generate_image_with_imagen (変更なし - Vertex AIを使用)
    def generate_image_with_imagen(prompt, project_id, region):
        """Vertex AI Imagen APIを呼び出し、画像バイトを返す関数"""
        try:
            vertexai.init(project=project_id, location=region)
            model = ImageGenerationModel.from_pretrained("imagegeneration@005") # モデルバージョン確認
            # st.write(f"DEBUG: Calling Imagen with prompt: {prompt[:100]}...")
            response = model.generate_images(prompt=prompt, number_of_images=1)
            # st.write("DEBUG: Imagen response received.")
            if response.images:
                image_bytes = response.images[0]._image_bytes
                # st.write(f"DEBUG: Image bytes length: {len(image_bytes)}")
                return image_bytes
            else:
                st.error(f"Imagen APIエラー: 画像データなし. Response: {response}")
                return None
        except Exception as e:
            st.error(f"Vertex AI Imagen APIエラー: {e}")
            if "permission denied" in str(e).lower() or "quota" in str(e).lower():
                 st.error("認証/割り当て量エラーの可能性。GCP設定確認要。")
            return None

    # add_text_to_image (変更なし)
    def add_text_to_image(image, text, position, font_path, font_size, text_color=(0, 0, 0, 255)):
        try:
            base = image.convert("RGBA"); txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
            try: font = ImageFont.truetype(font_path, font_size)
            except IOError: st.error(f"フォント未検出/読込不可: {font_path}"); return None
            except Exception as font_e: st.error(f"フォント読込エラー: {font_e}"); return None
            draw = ImageDraw.Draw(txt_layer); draw.text(position, text, font=font, fill=text_color)
            out = Image.alpha_composite(base, txt_layer); return out.convert("RGB")
        except Exception as e: st.error(f"テキスト描画エラー: {e}"); return None


    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o + Imagen Ver.)")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2)
    with col1:
        # ▼▼▼ フォーム内のボタンを st.form_submit_button に変更 ▼▼▼
        with st.form("input_form"): # clear_on_submit は外したまま
            st.subheader("1. 構成案のアップロード")
            uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])
            st.subheader("2. テキスト指示")
            impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
            details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ ..., B:見出し ...) ※改行して入力", height=200)
            st.subheader("3. 生成サイズ")
            st.info("注意: 現在、画像サイズはImagenのデフォルト(通常1024x1024)で生成されます。")
            # dalle_size selectbox はコメントアウト中

            # ★★★ ここを st.form_submit_button に変更 ★★★
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")
            # ★★★ ここを st.form_submit_button に変更 ★★★
        # ▲▲▲ フォームここまで ▲▲▲

        # 画像プレビュー
        if uploaded_file is not None:
             try:
                 image_preview = Image.open(uploaded_file)
                 st.image(image_preview, caption='アップロードされた構成案', use_column_width=True)
             except Exception as e: pass


    # --- ボタンが押された後の処理 ---
    # generate_button はフォーム送信時にTrueになる
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + Imagen)'):
                    # Step 1: レイアウト解析
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY, layout_analysis_prompt_text)
                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)
                        # Step 2: プロンプト生成
                        st.info("Step 2/3: 画像生成用プロンプトを生成中 (GPT-4o Text)...")
                        # サイズ指定はImagen側で無視されるため形式のみ影響
                        image_gen_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, "1024x1024", OPENAI_API_KEY, dalle_instruction_template_text)
                        if image_gen_prompt:
                             if "sorry" in image_gen_prompt.lower(): st.error("GPT-4oプロンプト生成失敗")
                             with st.expander("生成された画像生成プロンプト (GPT-4o)", expanded=True): st.text(image_gen_prompt)
                             if "sorry" not in image_gen_prompt.lower():
                                 # Step 3: 画像生成 (Imagen)
                                 st.info("Step 3/3: 画像を生成中 (Vertex AI Imagen)...")
                                 image_bytes = generate_image_with_imagen(image_gen_prompt, GCP_PROJECT_ID, GCP_REGION)
                                 if image_bytes:
                                      st.success("🎉 画像生成が完了しました！")
                                      st.subheader("生成されたラフ画像")
                                      # Step 4: 画像表示 & テキスト描画
                                      try:
                                           st.info("Step 4/4: テキスト描画処理中...")
                                           base_image = Image.open(BytesIO(image_bytes))
                                           # --- 固定テキスト描画テスト ---
                                           font_file_path = None
                                           try:
                                               script_dir = os.path.dirname(__file__)
                                               font_file_path = os.path.join(script_dir, "fonts", "NotoSansJP-Regular.ttf")
                                               if not os.path.exists(font_file_path): font_file_path = None
                                           except NameError:
                                               font_file_path = "fonts/NotoSansJP-Regular.ttf"
                                               if not os.path.exists(font_file_path): font_file_path = None

                                           if font_file_path:
                                                final_image = add_text_to_image(base_image, "テスト描画(Imagen) ABC", (30,30), font_file_path, 50, (0,0,255,255))
                                           else:
                                                st.warning("フォント未検出のためテキスト描画スキップ")
                                                final_image = base_image

                                           if final_image:
                                                st.image(final_image, caption='生成＋テキスト描画 結果 (Imagen)', use_column_width=True)
                                                st.balloons()
                                           else: st.error("テキスト描画エラー"); st.image(base_image, ...)

                                      except Exception as display_e:
                                           st.error(f"画像処理/表示エラー: {display_e}")
        else:
             with col1: st.warning("👈 画像アップロードと詳細指示を入力してください。")

# --- Error handling for failed prompt/secret loading ---
elif not prompts_loaded: st.error("プロンプト読込失敗。")
elif not secrets_ok: st.warning("Secrets関連で問題発生。")

