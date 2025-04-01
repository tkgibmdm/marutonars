# ↑ %%writefile app.py はGitHubには含めないでください！
import streamlit as st
# --- Set page config FIRST ---
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")

# --- Libraries ---
from openai import OpenAI
import requests
from PIL import Image, ImageDraw, ImageFont # Pillowのモジュールを追加
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
dalle_instruction_template_text = load_prompt("prompts/dalle_prompt_instruction_template.txt")
prompts_loaded = layout_analysis_prompt_text is not None and dalle_instruction_template_text is not None

# --- Load API Keys ---
secrets_ok = False
if prompts_loaded: # プロンプトが正常に読み込めたら次に進む
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Loaded but not used
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        if not GOOGLE_API_KEY or not OPENAI_API_KEY:
            st.error("エラー: Streamlit CloudのSecretsに必要なAPIキーが設定されていないか、値が空です。(GOOGLE_API_KEY, OPENAI_API_KEY)")
            st.stop()
        secrets_ok = True
    except KeyError as e:
        st.error(f"エラー: Streamlit CloudのSecretsにキー '{e}' が見つかりません。")
        st.stop()
    except Exception as e:
         st.error(f"Secrets読み込み中に予期せぬエラーが発生しました: {e}")
         st.stop()

# --- API Function Definitions ---
if secrets_ok and prompts_loaded:

    # ▼▼▼ GPT-4o (Vision) でレイアウト解析 ▼▼▼
    def analyze_layout_with_gpt4o(image_bytes, api_key, layout_prompt_text):
        """OpenAI GPT-4o APIを呼び出し、レイアウト情報を抽出する関数"""
        if not layout_prompt_text: st.error("レイアウト解析プロンプト未読込"); return None
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try:
                img = Image.open(BytesIO(image_bytes)); img_format = img.format
                if img_format == 'PNG': mime_type = "image/png"
                elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
                else: mime_type = "image/jpeg"
            except Exception: mime_type = "image/jpeg"
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": layout_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}]}]
            response = client.chat.completions.create(model="gpt-4o", messages=prompt_messages, max_tokens=1000)
            if response.choices and response.choices[0].message.content: return response.choices[0].message.content.strip()
            else: st.warning(f"GPT-4o応答なし(レイアウト解析). R: {response}"); return None
        except Exception as e: st.error(f"GPT-4o APIエラー(レイアウト解析): {e}"); return None
    # ▲▲▲ GPT-4o (Vision) でレイアウト解析 ▲▲▲

    # ▼▼▼ GPT-4o (Text) でDALL-Eプロンプト生成 ▼▼▼
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text):
        """OpenAI GPT-4o APIを呼び出し、DALL-E 3用プロンプトを生成する関数 (テキスト直接描画試行)"""
        if not dalle_instruction_template_text: st.error("DALL-E指示テンプレート未読込"); return None
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
    # ▲▲▲ GPT-4o (Text) でDALL-Eプロンプト生成 ▲▲▲

    def generate_image_with_dalle3(prompt, size, api_key):
        """DALL-E 3 APIを呼び出し、画像URLを返す関数"""
        try:
            client = OpenAI(api_key=api_key)
            response = client.images.generate(
                model="dall-e-3", prompt=prompt, size=size, quality="standard", style="vivid", n=1,
            )
            if response.data and len(response.data) > 0: return response.data[0].url
            else: st.error(f"DALL-E 3 APIエラー: URL取得失敗. R: {response}"); return None
        except Exception as e: st.error(f"DALL-E 3 APIエラー: {e}"); return None

    # ▼▼▼ テキスト描画関数 (変更なし) ▼▼▼
    def add_text_to_image(image, text, position, font_path, font_size, text_color=(0, 0, 0, 255)):
        """Pillowを使って画像にテキストを描画する関数"""
        try:
            base = image.convert("RGBA")
            txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError: st.error(f"フォント未検出/読込不可: {font_path}"); return None
            except Exception as font_e: st.error(f"フォント読込エラー: {font_e}"); return None
            draw = ImageDraw.Draw(txt_layer)
            draw.text(position, text, font=font, fill=text_color)
            out = Image.alpha_composite(base, txt_layer)
            return out.convert("RGB")
        except Exception as e: st.error(f"テキスト描画エラー: {e}"); return None
    # ▲▲▲ テキスト描画関数 ▲▲▲

    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o Ver.)")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    # --- Text Drawing Test Section Removed ---

    col1, col2 = st.columns(2)
    with col1:
        # フォーム (clear_on_submit は外したまま)
        with st.form("input_form"):
            st.subheader("1. 構成案のアップロード")
            uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])
            st.subheader("2. テキスト指示")
            impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
            details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ ..., B:見出し ...) ※改行して入力", height=200)
            st.subheader("3. 生成サイズ")
            dalle_size = st.selectbox("生成したい画像のサイズを選択 (DALL-E 3)",("1024x1024", "1792x1024", "1024x1792"), index=0)
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")

        # 画像プレビュー
        if uploaded_file is not None:
             try:
                 image_preview = Image.open(uploaded_file)
                 st.image(image_preview, caption='アップロードされた構成案', use_column_width=True)
             except Exception as e: pass


    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + DALL-E 3)'):
                    # Step 1: レイアウト解析
                    st.info("Step 1/3: 構成案画像を解析中...")
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY, layout_analysis_prompt_text)
                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)
                        # Step 2: DALL-E プロンプト生成
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中...")
                        dalle_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, dalle_size, OPENAI_API_KEY, dalle_instruction_template_text)
                        if dalle_prompt:
                             if "sorry" in dalle_prompt.lower(): st.error("GPT-4oプロンプト生成失敗")
                             with st.expander("生成されたDALL-Eプロンプト (GPT-4o)", expanded=True): st.text(dalle_prompt)
                             if "sorry" not in dalle_prompt.lower():
                                 # Step 3: 画像生成
                                 st.info("Step 3/3: DALL-E 3 画像生成中...")
                                 image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)
                                 if image_url:
                                      st.info("Step 4/4: テキスト描画処理中...") # ★変更
                                      st.subheader("生成されたラフ画像")
                                      # --- 画像表示 & テキスト描画テスト ---
                                      try:
                                           image_response = requests.get(image_url); image_response.raise_for_status()
                                           img_data = image_response.content
                                           base_image = Image.open(BytesIO(img_data)) # Pillowで開く

                                           # --- ▼▼▼ 固定テキスト描画テスト ▼▼▼ ---
                                           font_file_path = None
                                           try: # フォントパス取得
                                               script_dir = os.path.dirname(__file__)
                                               font_file_path = os.path.join(script_dir, "fonts", "NotoSansJP-Regular.ttf")
                                               if not os.path.exists(font_file_path): font_file_path = None
                                           except NameError: # フォールバック
                                               font_file_path = "fonts/NotoSansJP-Regular.ttf"
                                               if not os.path.exists(font_file_path): font_file_path = None

                                           if font_file_path:
                                                # add_text_to_image 関数を呼び出す
                                                final_image = add_text_to_image(
                                                    image=base_image,
                                                    text="テスト描画 ABC 123", # サンプルテキスト
                                                    position=(30, 30),       # サンプル座標 (左上)
                                                    font_path=font_file_path,
                                                    font_size=50,             # サンプルサイズ
                                                    text_color=(255, 0, 0, 255) # サンプル色 (赤)
                                                )
                                           else:
                                                st.warning("フォントファイルが見つからないため、テキストは描画されません。")
                                                final_image = base_image # 元画像を表示

                                           # --- ▲▲▲ 固定テキスト描画テスト ▲▲▲ ---

                                           if final_image:
                                                st.image(final_image, caption='生成＋テキスト描画 結果', use_column_width=True)
                                                st.success("🎉 処理完了！") # メッセージ変更
                                                st.balloons()
                                           else:
                                                st.error("テキスト描画処理でエラーが発生しました。")
                                                st.image(base_image, caption='生成された画像 (テキスト描画失敗)', use_column_width=True)

                                      except Exception as display_e:
                                           st.error(f"画像処理/表示エラー: {display_e}"); st.write(f"元画像URL: {image_url}")
        else:
             with col1: st.warning("👈 画像アップロードと詳細指示を入力してください。")

# --- Error handling for failed prompt/secret loading ---
elif not prompts_loaded: st.error("プロンプト読込失敗。")
elif not secrets_ok: st.warning("Secrets関連で問題発生。")

