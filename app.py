# ↑ この行はGitHubには含めないでください！ 以下を app.py の内容とします。
import streamlit as st
# --- Set page config FIRST ---
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")

# --- Libraries ---
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import time
import base64
import os # ファイル存在確認のため

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
        # GOOGLE_API_KEY は現在使わないが、Secrets設定はそのまま
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        if not GOOGLE_API_KEY or not OPENAI_API_KEY:
            st.error("エラー: APIキーが設定されていません。")
            st.stop()
        secrets_ok = True
    except KeyError as e:
        st.error(f"エラー: Secretsにキー '{e}' が見つかりません。")
        st.stop()
    except Exception as e:
         st.error(f"Secrets読み込みエラー: {e}")
         st.stop()

# --- API Function Definitions ---
# (APIキーとプロンプトが正常に読み込めた場合のみ定義・実行)
if secrets_ok and prompts_loaded:

    # ▼▼▼ GPT-4o (Vision) でレイアウト解析 (ファイルから読み込んだプロンプト使用) ▼▼▼
    def analyze_layout_with_gpt4o(image_bytes, api_key, layout_prompt_text): # 引数にプロンプト追加
        """OpenAI GPT-4o APIを呼び出し、レイアウト情報を抽出する関数"""
        if not layout_prompt_text: # 関数呼び出し前にもチェック
             st.error("レイアウト解析プロンプトが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try: # MIMEタイプ判定
                img_format = Image.open(BytesIO(image_bytes)).format
                if img_format == 'PNG': mime_type = "image/png"
                elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
                else: mime_type = "image/jpeg"
            except Exception: mime_type = "image/jpeg"

            prompt_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": layout_prompt_text}, # ファイルから読み込んだプロンプトを使用
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]}
            ]
            response = client.chat.completions.create(
                model="gpt-4o", messages=prompt_messages, max_tokens=1000
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（レイアウト解析）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (レイアウト解析): {e}")
            return None
    # ▲▲▲ GPT-4o (Vision) でレイアウト解析 ▲▲▲

    # ▼▼▼ GPT-4o (Text) でDALL-Eプロンプト生成 (ファイルから読み込んだテンプレート使用) ▼▼▼
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text): # 引数にテンプレート追加
        """OpenAI GPT-4o APIを呼び出し、DALL-E 3用プロンプトを生成する関数"""
        if not dalle_instruction_template_text: # 関数呼び出し前にもチェック
             st.error("DALL-Eプロンプト生成指示テンプレートが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            # --- テンプレートに動的な値を埋め込む ---
            prompt_generation_instruction = dalle_instruction_template_text.format(
                size=size,
                layout_info=layout_info,
                impression=impression,
                details=details
            )
            # --- 埋め込みここまで ---

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_generation_instruction}],
                max_tokens=1500
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（プロンプト生成）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (プロンプト生成): {e}")
            return None
    # ▲▲▲ GPT-4o (Text) でDALL-Eプロンプト生成 ▲▲▲

    def generate_image_with_dalle3(prompt, size, api_key):
        """DALL-E 3 APIを呼び出し、画像URLを返す関数"""
        # --- この関数は変更なし ---
        try:
            client = OpenAI(api_key=api_key)
            response = client.images.generate(
                model="dall-e-3", prompt=prompt, size=size,
                quality="standard", style="vivid", n=1,
            )
            if response.data and len(response.data) > 0: return response.data[0].url
            else:
                st.error("DALL-E 3 APIエラー: レスポンスから画像URLを取得できませんでした。")
                return None
        except Exception as e:
            st.error(f"DALL-E 3 APIエラー: {e}")
            return None

    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o Ver.)")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2)
    with col1:
        # --- Form definition (Unchanged) ---
        with st.form("input_form", clear_on_submit=True):
            st.subheader("1. 構成案のアップロード")
            uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])
            st.subheader("2. テキスト指示")
            impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
            details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ ..., B:見出し ...) ※改行して入力", height=200)
            st.subheader("3. 生成サイズ")
            dalle_size = st.selectbox("生成したい画像のサイズを選択 (DALL-E 3)",("1024x1024", "1792x1024", "1024x1792"), index=0)
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")

        # Image preview (Unchanged)
        if uploaded_file is not None:
             try: image = Image.open(uploaded_file); st.image(image, ...)
             except: pass

    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + DALL-E 3)'):
                    # Step 1: Calls analyze_layout_with_gpt4o (Uses prompt loaded from file)
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    # ファイルから読み込んだプロンプトテキストを渡す
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY, layout_analysis_prompt_text)
                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)
                        # Step 2: Calls generate_dalle_prompt_with_gpt4o (Uses template loaded from file)
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (GPT-4o Text)...")
                        # ファイルから読み込んだテンプレートテキストを渡す
                        dalle_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, dalle_size, OPENAI_API_KEY, dalle_instruction_template_text)
                        if dalle_prompt:
                             with st.expander("生成されたDALL-Eプロンプト (GPT-4o)", expanded=True): st.text(dalle_prompt)
                             # Step 3: Calls generate_image_with_dalle3
                             st.info("Step 3/3: 画像を生成中 (DALL-E 3)...")
                             image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)
                             if image_url:
                                  # ... (Display image logic) ...
                                  st.success("🎉 画像生成が完了しました！")
                                  st.subheader("生成されたラフ画像")
                                  try:
                                       image_response = requests.get(image_url); image_response.raise_for_status()
                                       img_data = image_response.content; st.image(img_data, ...)
                                       st.balloons()
                                  except Exception as download_e:
                                       st.error(f"画像表示エラー: {download_e}"); st.write(f"URL: {image_url}")
        else:
            st.warning("👈 画像アップロードと詳細指示を入力してください。")

# --- Error handling for failed prompt/secret loading ---
elif not prompts_loaded:
     st.error("プロンプトファイルの読み込みに失敗したため、アプリを起動できません。GitHubリポジトリ内の 'prompts' フォルダとファイルを確認してください。")
elif not secrets_ok:
     st.warning("アプリの初期化中にSecrets関連で問題が発生したため、UIを表示できません。")
