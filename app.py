# ↑ この行はGitHubには含めないでください！ 以下を app.py の内容とします。
import streamlit as st
# --- Set page config FIRST ---
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")

# --- Libraries ---
# import google.generativeai as genai # Not used
from openai import OpenAI # Used for Step 1 & 3 (via client), and Step 2
import requests
from PIL import Image
from io import BytesIO
import time
import base64 # GPT-4o Vision用に必要
import os # ファイルパス判定用 (念のため)

# --- Load API Keys ---
secrets_ok = False
try:
    # GOOGLE_API_KEY は現在使いませんが、Secretsに設定されているか一応確認
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # こちらを主に使用
    if not GOOGLE_API_KEY or not OPENAI_API_KEY:
        st.error("エラー: Streamlit CloudのSecretsに必要なAPIキーが設定されていないか、値が空です。(GOOGLE_API_KEY, OPENAI_API_KEY)")
        st.stop()
    secrets_ok = True
except KeyError as e:
    st.error(f"エラー: Streamlit CloudのSecretsにキー '{e}' が見つかりません。")
    st.stop()
except Exception as e:
     st.error(f"Secrets読み込み中に予期せぬエラー: {e}")
     st.stop()

# --- API Function Definitions ---
if secrets_ok:

    # ▼▼▼ GPT-4o (Vision) でレイアウト解析 ▼▼▼
    def analyze_layout_with_gpt4o(image_bytes, api_key):
        """OpenAI GPT-4o APIを呼び出し、レイアウト情報を抽出する関数"""
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try:
                img_format = Image.open(BytesIO(image_bytes)).format
                if img_format == 'PNG': mime_type = "image/png"
                elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
                else: mime_type = "image/jpeg"
            except Exception: mime_type = "image/jpeg"

            prompt_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": """
                    この画像はウェブバナー広告の構成案（ラフスケッチ）です。
                    画像に含まれる主要な要素（例えば、ロゴ、見出しテキスト、本文テキスト、画像エリア、ボタンなど）を特定してください。
                    そして、それぞれの要素について、以下の情報をリスト形式で記述してください。
                    - 要素の種類（例: テキスト、画像、ロゴ、ボタン）
                    - おおよその位置（例: 上部中央、左下、右側3分の1）
                    - 要素内に読み取れるテキスト（もしあれば）
                    - 簡単な形状や特徴（例: 横長の長方形、円形）
                    """},
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

    # ▼▼▼ GPT-4o (Text) でDALL-Eプロンプト生成 (テキスト直接描画指示バージョン) ▼▼▼
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key):
        """OpenAI GPT-4o APIを呼び出し、DALL-E 3用プロンプトを生成する関数 (テキスト直接描画試行)"""
        try:
            client = OpenAI(api_key=api_key)
            # ▼▼▼ テキスト直接描画を試みる指示に変更 ▼▼▼
            prompt_generation_instruction = f"""
            You are an expert prompt engineer for the DALL-E 3 image generation model.
            Based on the following "Layout Information" (extracted from a rough sketch by GPT-4o) and "Specific Element Instructions", create a detailed and effective **English prompt** for DALL-E 3 to generate a banner ad image.

            **Target Image Specifications:**
            * Size: {size}
            * Goal: Accurately reflect the specified elements, their approximate layout, and the overall style.

            **Instructions for DALL-E Prompt Generation:**
            1.  **Prioritize Specific Instructions:** Use the text and descriptions from "Specific Element Instructions". Ignore any descriptive text extracted from the sketch in "Layout Information" if it conflicts.
            2.  **Describe Layout Simply:** For element positioning, use simple, clear terms based on the "Layout Information" (e.g., "top left corner", "upper center area", "bottom right corner"). Avoid overly complex relative positioning instructions. Describe the placement of each key element clearly.
            3.  **Content Details:** Include the specific image descriptions provided for each element (A, B, C, etc.).
            4.  **Text Content:** Include the specific text content provided for each text element (headline, body, button text, date, etc.) **directly in the description for DALL-E 3**. Use quotes to indicate the exact text. Acknowledge that DALL-E 3 might not render text perfectly, but include the requested text anyway. Example for a button: "In the bottom right corner, include a rectangular button shape containing the text 'Register for Free Here'." Example for a headline: "In the upper center area, display the headline text 'Master AI-Powered Design'."
            5.  **Overall Style:** Incorporate the "Overall atmosphere/style" instructions into the prompt, describing the visual theme, colors, mood, and background.
            6.  **Negative Constraints:** Explicitly add instructions to AVOID generating unspecified elements. Example: "Do not include any extra people, user interface elements, or objects not mentioned in the element descriptions."
            7.  **Output Format:** Output ONLY the final English prompt for DALL-E 3, without any introductory phrases, explanations, or markdown formatting.

            # Layout Information (from sketch analysis by GPT-4o)
            ```
            {layout_info}
            ```

            # Specific Element Instructions (from user input)
            ```
            - Overall atmosphere/style: {impression}
            {details}
            ```

            Generate the DALL-E 3 prompt now.
            """
            # ▲▲▲ テキスト直接描画を試みる指示に変更 ▲▲▲

            response = client.chat.completions.create(
                model="gpt-4o", # プロンプト生成もGPT-4oで
                messages=[{"role": "user", "content": prompt_generation_instruction}],
                max_tokens=1500 # Adjust as needed
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（プロンプト生成）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (プロンプト生成): {e}")
            return None
    # ▲▲▲ GPT-4o (Text) でDALL-Eプロンプト生成 (テキスト直接描画指示バージョン) ▲▲▲

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
        # --- フォーム定義 (変更なし) ---
        with st.form("input_form", clear_on_submit=True):
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
                 st.image(image, caption='アップロードされた構成案 (クリア後は消えます)', use_column_width=True)
             except Exception as e:
                 st.error(f"画像プレビューエラー: {e}")


    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + DALL-E 3)'):
                    # Step 1: GPT-4o でレイアウト解析
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY)

                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)
                        # Step 2: GPT-4o で DALL-E プロンプト生成
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (GPT-4o Text)...")
                        dalle_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, dalle_size, OPENAI_API_KEY)

                        if dalle_prompt:
                             with st.expander("生成されたDALL-Eプロンプト (GPT-4o)", expanded=True): st.text(dalle_prompt)
                             # Step 3: DALL-E 3 で画像生成
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
        else:
            st.warning("👈 画像のアップロードと詳細指示を入力してからボタンを押してください。")

else: # if secrets_ok is False
    st.warning("アプリの初期化中にSecrets関連で問題が発生したため、UIを表示できません。")
