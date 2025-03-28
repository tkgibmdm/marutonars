import streamlit as st

# ▼▼▼ set_page_config をここに移動 ▼▼▼
st.set_page_config(page_title="AIバナーラフ生成", layout="wide")
# ▲▲▲ set_page_config をここに移動 ▲▲▲

# --- 他のライブラリ import (ここでもOK) ---
# import google.generativeai as genai
# ...

st.write("DEBUG: Script started. Importing libraries...") # Debug 0
st.write("DEBUG: Attempting to load secrets...") # Debug 1
# ... (以下、Secrets読み込み処理) ...

# --- APIキー読み込み ---
st.write("DEBUG: Attempting to load secrets...") # Debug 1
GOOGLE_API_KEY = None # 初期化
OPENAI_API_KEY = None # 初期化
secrets_ok = False    # 読み込み成功フラグ

try:
    st.write("DEBUG: Inside try block. Accessing GOOGLE_API_KEY...") # Debug 2
    # Streamlit CloudのSecretsから読み込む
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    st.write("DEBUG: GOOGLE_API_KEY access attempted.") # Debug 3

    st.write("DEBUG: Accessing OPENAI_API_KEY...") # Debug 4
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.write("DEBUG: OPENAI_API_KEY access attempted.") # Debug 5

    st.write("DEBUG: Checking if keys are empty...") # Debug 6
    if not GOOGLE_API_KEY or not OPENAI_API_KEY:
        st.error("エラー: Streamlit CloudのSecretsにAPIキーが設定されていないか、値が空です。")
        st.write("DEBUG: Stopping app because keys are missing or empty.") # Debug 7
        st.stop() # 値が空ならここで停止

    st.write("DEBUG: API Keys seem valid (not empty).") # Debug 8
    secrets_ok = True # ここまで来たらOK

except KeyError as e:
    # 指定したキー名が存在しない場合
    st.error(f"エラー: Streamlit CloudのSecretsにキー '{e}' が見つかりません。名前が正しいか確認してください。")
    st.write(f"DEBUG: Stopping app due to KeyError: {e}") # Debug 9
    st.stop() # ここで停止
except Exception as e:
     # その他の予期せぬエラー
     st.error(f"Secrets読み込み中に予期せぬエラーが発生しました: {e}")
     st.write(f"DEBUG: Stopping app due to unexpected secret error: {e}") # Debug 10
     st.stop() # ここで停止

st.write("DEBUG: Secret loading block finished successfully. Proceeding to UI...") # Debug 11

# --- ここから下のUI定義や関数定義は、Secretsが正常に読み込めた場合のみ実行 ---
if secrets_ok:
    # --- 必要なライブラリをここでインポート ---
    # (Secrets読み込み後にインポートすることで、エラー箇所を特定しやすくする)
    import google.generativeai as genai
    from openai import OpenAI
    import requests
    from PIL import Image
    from io import BytesIO

    # --- API呼び出し関数定義 ---
    # (前回と同じ関数定義をここに記述)
    def analyze_layout_with_gemini(image_bytes, api_key):
        """Gemini Vision APIを呼び出し、レイアウト情報を抽出する関数"""
        try:
            genai.configure(api_key=api_key)
            img = Image.open(BytesIO(image_bytes))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = """
            この画像はウェブバナー広告の構成案（ラフスケッチ）です。
            画像に含まれる主要な要素を特定し、それぞれの種類、おおよその位置、
            読み取れるテキスト（もしあれば）、形状をリスト形式で記述してください。
            """
            response = model.generate_content([prompt, img])
            if response.parts: return response.text
            else:
                st.warning(f"Gemini Vision応答なし. Feedback: {response.prompt_feedback}")
                return None
        except Exception as e:
            st.error(f"Gemini Vision APIエラー: {e}")
            return None

    def generate_dalle_prompt_with_gemini(layout_info, impression, details, size, api_key):
        """Gemini Text APIを呼び出し、DALL-E 3用プロンプトを生成する関数"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt_generation_prompt = f"""
            以下の「レイアウト情報」と「要素の具体的な内容指示」に基づいて、画像生成AIであるDALL-E 3で画像を生成するための、非常に詳細で具体的な英語のプロンプトを作成してください。
            生成する画像のサイズは {size} です。レイアウト情報と要素内容指示を正確に反映し、全体の雰囲気・スタイルも考慮してください。DALL-E 3が理解しやすいように、自然言語で詳しく記述してください。
            特に、レイアウト情報にあるテキスト要素の内容は無視し、「要素の具体的な内容指示」にあるテキストを使用してください。
            # レイアウト情報
            ```{layout_info}```
            # 要素の具体的な内容指示
            ```- 全体の雰囲気・スタイル: {impression}\n{details}```
            英語のプロンプトのみを出力してください。余計な前置きや後書きは不要です。
            """
            response = model.generate_content(prompt_generation_prompt)
            if response.parts:
                 generated_text = response.text.strip()
                 if generated_text.startswith("```") and generated_text.endswith("```"):
                      lines = generated_text.split('\n')
                      if len(lines) > 2:
                           if lines[0].startswith("```"): lines.pop(0)
                           if lines[-1] == "```": lines.pop()
                           generated_text = "\n".join(lines).strip()
                 return generated_text
            else:
                 st.warning(f"Gemini Text応答なし (プロンプト生成). Feedback: {response.prompt_feedback}")
                 return None
        except Exception as e:
            st.error(f"Gemini Text APIエラー (プロンプト生成): {e}")
            return None

    def generate_image_with_dalle3(prompt, size, api_key):
        """DALL-E 3 APIを呼び出し、画像URLを返す関数"""
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

    st.title("🤖 AIバナーラフ生成プロトタイプ")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")
    st.write("--- Debug Info ---")
    st.write(f"Google Key Loaded: {'Yes' if GOOGLE_API_KEY else 'No'}")
    st.write(f"OpenAI Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
    st.write("--- End Debug Info ---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 構成案のアップロード")
        uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])
        uploaded_image_preview = None
        if uploaded_file is not None:
            try:
                uploaded_image_preview = Image.open(uploaded_file)
                st.image(uploaded_image_preview, caption='アップロードされた構成案', use_column_width=True)
            except Exception as e:
                st.error(f"画像ファイルの読み込み中にエラーが発生しました: {e}")
                uploaded_file = None
        st.subheader("2. テキスト指示")
        impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
        details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ KAWAI DESIGN, B:見出し AI駆動...) ※改行して入力してください", height=200)
        st.subheader("3. 生成サイズ")
        dalle_size = st.selectbox("生成したい画像のサイズを選択 (DALL-E 3)",("1024x1024", "1792x1024", "1024x1792"), index=0)
        generate_button = st.button("🖼️ ラフ画像を生成する", type="primary")

    # --- ボタンが押された後の処理 ---
    if generate_button:
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です...しばらくお待ちください...'):
                    # Step 1: レイアウト解析
                    st.info("Step 1/3: 構成案画像を解析中 (Gemini Vision)...")
                    layout_info = analyze_layout_with_gemini(layout_image_bytes, GOOGLE_API_KEY)
                    if layout_info:
                        with st.expander("レイアウト解析結果 (Gemini Vision)", expanded=False): st.text(layout_info)
                        # Step 2: DALL-E プロンプト生成
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (Gemini Text)...")
                        dalle_prompt = generate_dalle_prompt_with_gemini(layout_info, impression_text, details_text, dalle_size, GOOGLE_API_KEY)
                        if dalle_prompt:
                             with st.expander("生成されたDALL-Eプロンプト (Gemini Text)", expanded=False): st.text(dalle_prompt)
                             # Step 3: 画像生成
                             st.info("Step 3/3: 画像を生成中 (DALL-E 3)...")
                             image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)
                             if image_url:
                                  st.success("🎉 画像生成が完了しました！")
                                  st.subheader("生成されたラフ画像")
                                  # Step 4: 画像表示
                                  try:
                                       image_response = requests.get(image_url)
                                       image_response.raise_for_status()
                                       img_data = image_response.content
                                       st.image(img_data, caption='生成されたラフ画像', use_column_width=True)
                                       st.balloons()
                                  except Exception as download_e:
                                       st.error(f"生成された画像のダウンロード/表示エラー: {download_e}")
                                       st.write(f"画像URLはこちら: {image_url}")
        else:
            st.warning("👈 構成案の画像アップロードと、各要素の詳細指示を入力してからボタンを押してください。")

else: # if secrets_ok is False
    st.warning("アプリの初期化中にSecrets関連で問題が発生したため、UIを表示できません。ログを確認してください。")
