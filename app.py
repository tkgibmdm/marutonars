# ↑ %%writefile app.py はGitHubには含めないでください！
import streamlit as st
# --- Set page config FIRST ---
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")

# --- Libraries ---
from openai import OpenAI # Step 1, 2 で使用
import vertexai # ★ Vertex AI をインポート
from vertexai.preview.vision_models import ImageGenerationModel # ★ Imagenモデルをインポート
# import requests # URLダウンロード不要のためコメントアウトまたは削除
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
        # ★ GCPプロジェクトIDとリージョンをSecretsから読み込む
        GCP_PROJECT_ID = st.secrets["GCP_PROJECT_ID"]
        GCP_REGION = st.secrets["GCP_REGION"]
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # GPT-4o用に必要
        # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Vertex AIでは通常不要

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
        # ... (関数定義は前回と同じ) ...
        if not layout_prompt_text: st.error("レイアウト解析プロンプト未読込"); return None
        try:
            client = OpenAI(api_key=api_key)
            # ... (base64, mime type, API call) ...
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
    # 関数名はそのまま使うが、生成するのはImagen向けプロンプトになる点に注意
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text):
        """OpenAI GPT-4o APIを呼び出し、画像生成用プロンプトを生成する関数"""
        # ... (関数定義は前回と同じ、テンプレートに従ってプロンプトを生成) ...
        if not dalle_instruction_template_text: st.error("画像生成指示テンプレート未読込"); return None
        try:
            client = OpenAI(api_key=api_key)
            prompt_generation_instruction = dalle_instruction_template_text.format(
                 size=size, layout_info=layout_info, impression=impression, details=details
            ) # Note: size format might need adjustment for Imagen later
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt_generation_instruction}], max_tokens=1500
            )
            if response.choices and response.choices[0].message.content: return response.choices[0].message.content.strip()
            else: st.warning(f"GPT-4o応答なし(プロンプト生成). R: {response}"); return None
        except Exception as e: st.error(f"GPT-4o APIエラー(プロンプト生成): {e}"); return None


    # ▼▼▼ DALL-E 3 の代わりに Imagen を使う関数 ▼▼▼
    def generate_image_with_imagen(prompt, project_id, region):
        """Vertex AI Imagen APIを呼び出し、画像バイトを返す関数"""
        try:
            # Vertex AIを初期化 (毎回呼んでも問題ないはず)
            vertexai.init(project=project_id, location=region)
            # 利用可能なモデルを確認・指定 (例: imagegeneration@005)
            # 最新版はドキュメントで確認推奨
            model = ImageGenerationModel.from_pretrained("imagegeneration@005")
            st.write(f"DEBUG: Calling Imagen with prompt: {prompt[:100]}...") # Debug

            # 画像生成を実行 (サイズ指定は一旦省略してデフォルトに任せる)
            response = model.generate_images(
                prompt=prompt,
                number_of_images=1
                # size や aspect_ratio パラメータは後で試す
            )
            st.write("DEBUG: Imagen response received.") # Debug

            if response.images:
                # 画像データをバイト形式で取得
                image_bytes = response.images[0]._image_bytes
                st.write(f"DEBUG: Image bytes length: {len(image_bytes)}") # Debug
                return image_bytes
            else:
                st.error(f"Imagen APIエラー: 画像データがレスポンスに含まれていません。Response: {response}")
                return None
        except Exception as e:
            st.error(f"Vertex AI Imagen APIエラー: {e}")
            # 認証エラーの場合、ADCやサービスアカウント設定を確認するよう促すメッセージを追加しても良い
            if "permission denied" in str(e).lower() or "quota" in str(e).lower():
                 st.error("認証エラーまたは割り当て量超過の可能性があります。GCPプロジェクトのVertex AI API設定や認証情報を確認してください。")
            return None
    # ▲▲▲ DALL-E 3 の代わりに Imagen を使う関数 ▲▲▲

    # add_text_to_image (変更なし)
    def add_text_to_image(image, text, position, font_path, font_size, text_color=(0, 0, 0, 255)):
        # ... (関数定義は前回と同じ) ...
        try:
            base = image.convert("RGBA"); txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
            try: font = ImageFont.truetype(font_path, font_size)
            except IOError: st.error(f"フォント未検出/読込不可: {font_path}"); return None
            except Exception as font_e: st.error(f"フォント読込エラー: {font_e}"); return None
            draw = ImageDraw.Draw(txt_layer); draw.text(position, text, font=font, fill=text_color)
            out = Image.alpha_composite(base, txt_layer); return out.convert("RGB")
        except Exception as e: st.error(f"テキスト描画エラー: {e}"); return None


    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o + Imagen Ver.)") # タイトル変更
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2)
    with col1:
        # フォーム (clear_on_submit は外したまま)
        with st.form("input_form"):
            st.subheader("1. 構成案のアップロード")
            uploaded_file = st.file_uploader(...) # Ellipsis for brevity
            st.subheader("2. テキスト指示")
            impression_text = st.text_input(...)
            details_text = st.text_area(...)
            st.subheader("3. 生成サイズ")
            # ★ Imagenのサイズ指定は複雑なため、一旦UIからは削除または無視する
            #    デフォルトサイズ (1024x1024?) で生成される
            # dalle_size = st.selectbox(...) # コメントアウトまたは削除
            st.info("注意: 現在、画像サイズはImagenのデフォルト(通常1024x1024)で生成されます。")
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")

        if uploaded_file is not None:
             try: image = Image.open(uploaded_file); st.image(image, ...)
             except: pass

    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                # スピナーのテキストも変更
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + Imagen)'):
                    # Step 1: レイアウト解析 (GPT-4o) - 変更なし
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY, layout_analysis_prompt_text)
                    if layout_info:
                        with st.expander(...): st.text(layout_info)
                        # Step 2: プロンプト生成 (GPT-4o) - 変更なし
                        # 注意: 生成されるプロンプトはImagen向けに最適化されていない可能性あり
                        st.info("Step 2/3: 画像生成用プロンプトを生成中 (GPT-4o Text)...")
                        # サイズ指定はImagen側で無視されるため、ここで渡す値は形式のみ影響
                        image_gen_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, "1024x1024", OPENAI_API_KEY, dalle_instruction_template_text) # 変数名変更
                        if image_gen_prompt:
                             if "sorry" in image_gen_prompt.lower(): st.error("GPT-4oプロンプト生成失敗")
                             with st.expander("生成された画像生成プロンプト (GPT-4o)", expanded=True): st.text(image_gen_prompt)
                             if "sorry" not in image_gen_prompt.lower():
                                 # ▼▼▼ Step 3: 画像生成 (Imagen) ▼▼▼
                                 st.info("Step 3/3: 画像を生成中 (Vertex AI Imagen)...")
                                 # GCP Project ID と Region を渡す
                                 image_bytes = generate_image_with_imagen(image_gen_prompt, GCP_PROJECT_ID, GCP_REGION)
                                 # ▲▲▲ Step 3: 画像生成 (Imagen) ▲▲▲

                                 if image_bytes: # Imagenはバイト列を返す
                                      st.success("🎉 画像生成が完了しました！")
                                      st.subheader("生成されたラフ画像")
                                      # --- Step 4: 画像表示 & テキスト描画 ---
                                      try:
                                           st.info("Step 4/4: テキスト描画処理中...")
                                           # バイトデータからPillowイメージを開く
                                           base_image = Image.open(BytesIO(image_bytes))

                                           # --- 固定テキスト描画テスト ---
                                           font_file_path = None
                                           try: # フォントパス取得
                                               script_dir = os.path.dirname(__file__)
                                               font_file_path = os.path.join(script_dir, "fonts", "NotoSansJP-Regular.ttf")
                                               if not os.path.exists(font_file_path): font_file_path = None
                                           except NameError: font_file_path = "fonts/NotoSansJP-Regular.ttf"; # Fallback
                                           if not os.path.exists(font_file_path): font_file_path = None

                                           if font_file_path:
                                                final_image = add_text_to_image(base_image, "テスト描画(Imagen) ABC", (30,30), font_file_path, 50, (0,0,255,255)) # 色変更
                                           else: final_image = base_image

                                           if final_image:
                                                st.image(final_image, caption='生成＋テキスト描画 結果 (Imagen)', use_column_width=True)
                                                st.balloons()
                                           else: st.error("テキスト描画エラー"); st.image(base_image, ...)

                                      except Exception as display_e:
                                           st.error(f"画像処理/表示エラー: {display_e}")
                                           # No URL to display for Imagen bytes
        else:
             with col1: st.warning("👈 画像アップロードと詳細指示を入力してください。")

# --- Error handling for failed prompt/secret loading ---
elif not prompts_loaded: st.error("プロンプト読込失敗。")
elif not secrets_ok: st.warning("Secrets関連で問題発生。")
