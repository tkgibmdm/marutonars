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
if prompts_loaded:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Loaded but not used
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
if secrets_ok and prompts_loaded:

    def analyze_layout_with_gpt4o(image_bytes, api_key, layout_prompt_text):
        """OpenAI GPT-4o APIを呼び出し、レイアウト情報を抽出する関数"""
        if not layout_prompt_text:
             st.error("レイアウト解析プロンプトが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try:
                img_format = Image.open(BytesIO(image_bytes)).format
                if img_format == 'PNG': mime_type = "image/png"
                elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
                else: mime_type = "image/jpeg"
            except Exception: mime_type = "image/jpeg"
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": layout_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}]}]
            response = client.chat.completions.create(model="gpt-4o", messages=prompt_messages, max_tokens=1000)
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（レイアウト解析）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (レイアウト解析): {e}")
            return None

    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text):
        """OpenAI GPT-4o APIを呼び出し、DALL-E 3用プロンプトを生成する関数 (テキスト直接描画試行)"""
        if not dalle_instruction_template_text:
             st.error("DALL-Eプロンプト生成指示テンプレートが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            # --- テンプレートに動的な値を埋め込む ---
            prompt_generation_instruction = dalle_instruction_template_text.format(
                size=size, layout_info=layout_info, impression=impression, details=details
            )
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt_generation_instruction}], max_tokens=1500
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（プロンプト生成）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (プロンプト生成): {e}")
            return None

    def generate_image_with_dalle3(prompt, size, api_key):
        """DALL-E 3 APIを呼び出し、画像URLを返す関数"""
        try:
            client = OpenAI(api_key=api_key)
            response = client.images.generate(
                model="dall-e-3", prompt=prompt, size=size, quality="standard", style="vivid", n=1,
            )
            if response.data and len(response.data) > 0: return response.data[0].url
            else:
                st.error(f"DALL-E 3 APIエラー: レスポンスから画像URLを取得できませんでした。Response: {response}")
                return None
        except Exception as e:
            st.error(f"DALL-E 3 APIエラー: {e}")
            return None

    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o Ver.)")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2)
    with col1:
        # ▼▼▼ デバッグのため clear_on_submit=True を一時的に削除 ▼▼▼
        with st.form("input_form"): # clear_on_submit=True を削除
        # ▲▲▲ デバッグのため clear_on_submit=True を一時的に削除 ▲▲▲
            st.subheader("1. 構成案のアップロード")
            uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])
            st.subheader("2. テキスト指示")
            impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
            details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ ..., B:見出し ...) ※改行して入力", height=200)
            st.subheader("3. 生成サイズ")
            dalle_size = st.selectbox("生成したい画像のサイズを選択 (DALL-E 3)",("1024x1024", "1792x1024", "1024x1792"), index=0)
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")

        # 画像プレビュー (変更なし)
        if uploaded_file is not None:
             try:
                 image = Image.open(uploaded_file)
                 st.image(image, caption='アップロードされた構成案', use_column_width=True)
             except Exception as e:
                 st.error(f"画像プレビューエラー: {e}")


    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            # Since clear_on_submit is False, uploaded_file should persist
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + DALL-E 3)'):
                    # Step 1: レイアウト解析
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY)

                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)

                        # ▼▼▼ デバッグプリント追加 ▼▼▼
                        st.write("--- DEBUG INFO for Step 2 ---")
                        st.write(f"Layout Info Provided (type: {type(layout_info)}):")
                        st.text(f"{layout_info[:500]}...") # 長いので一部表示
                        st.write(f"Impression Text Provided (type: {type(impression_text)}): {impression_text}")
                        st.write(f"Details Text Provided (type: {type(details_text)}, empty: {not details_text}):")
                        st.text(f"{details_text[:500]}...") # 長いので一部表示
                        st.write(f"DALL-E Size Provided: {dalle_size}")
                        st.write("--- END DEBUG INFO ---")
                        # ▲▲▲ デバッグプリント追加 ▲▲▲

                        # Step 2: DALL-E プロンプト生成
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (GPT-4o Text)...")
                        dalle_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, dalle_size, OPENAI_API_KEY)

                        if dalle_prompt:
                             # Check if dalle_prompt contains the "I'm sorry..." message
                             if "I'm sorry, but I need the specific" in dalle_prompt:
                                  st.error("GPT-4oがプロンプト生成に必要な情報を得られなかったようです。（上記DEBUG INFOを確認してください）")
                             with st.expander("生成されたDALL-Eプロンプト (GPT-4o)", expanded=True): st.text(dalle_prompt)

                             # Only proceed if prompt generation didn't result in the known error message
                             if "I'm sorry, but I need the specific" not in dalle_prompt:
                                 # Step 3: 画像生成
                                 st.info("Step 3/3: 画像を生成中 (DALL-E 3)...")
                                 image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)

                                 if image_url:
                                      st.success("🎉 画像生成が完了しました！")
                                      st.subheader("生成されたラフ画像")
                                      # Step 4: 画像表示
                                      try:
                                           image_response = requests.get(image_url); image_response.raise_for_status()
                                           img_data = image_response.content; st.image(img_data, caption='生成されたラフ画像', use_column_width=True)
                                           st.balloons()
                                      except Exception as download_e:
                                           st.error(f"画像ダウンロード/表示エラー: {download_e}"); st.write(f"画像URL: {image_url}")
                        # (else: GPT-4o prompt gen error already shown by the function)
                    # (else: GPT-4o layout analysis error already shown by the function)
        else:
            st.warning("👈 画像アップロードと詳細指示を入力してください。")

elif not prompts_loaded:
     st.error("プロンプトファイルの読み込みに失敗しました。")
elif not secrets_ok:
     st.warning("Secrets関連で問題が発生しました。")
