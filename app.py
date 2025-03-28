import streamlit as st
# --- 他のライブラリ import ---
import google.generativeai as genai
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import time

# --- APIキー読み込み ---
# (ここは前回のデバッグコードから、DEBUG用のst.writeを除いた形に戻すのが良いでしょう)
secrets_ok = False
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not GOOGLE_API_KEY or not OPENAI_API_KEY:
        st.error("エラー: Streamlit CloudのSecretsにAPIキーが設定されていないか、値が空です。")
        st.stop()
    secrets_ok = True
except KeyError as e:
    st.error(f"エラー: Streamlit CloudのSecretsにキー '{e}' が見つかりません。")
    st.stop()
except Exception as e:
     st.error(f"Secrets読み込み中に予期せぬエラーが発生しました: {e}")
     st.stop()

# --- API呼び出し関数定義 ---
# (ここは前回と同じ)
def analyze_layout_with_gemini(image_bytes, api_key):
    # ... (関数の中身は省略) ...
    try: # エラーハンドリング例
        # ... API呼び出し ...
        if response.parts: return response.text
        else: return None
    except Exception as e: return None

def generate_dalle_prompt_with_gemini(layout_info, impression, details, size, api_key):
    # ... (関数の中身は省略) ...
    try: # エラーハンドリング例
        # ... API呼び出し ...
        if response.parts: return response.text.strip() # 簡単な整形
        else: return None
    except Exception as e: return None

def generate_image_with_dalle3(prompt, size, api_key):
    # ... (関数の中身は省略) ...
    try: # エラーハンドリング例
        # ... API呼び出し ...
        if response.data and len(response.data) > 0: return response.data[0].url
        else: return None
    except Exception as e: return None


# --- Streamlit App Main UI ---
if secrets_ok: # Secretsが読み込めたらUIを表示
    st.set_page_config(page_title="AIバナーラフ生成", layout="wide")
    st.title("🤖 AIバナーラフ生成プロトタイプ")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2) # 画面を2カラムに分割

    # ▼▼▼ 左カラムの入力をフォームで囲む ▼▼▼
    with col1:
        # フォームを作成 (キーを設定し、送信後にクリアする)
        with st.form("input_form", clear_on_submit=True):
            st.subheader("1. 構成案のアップロード")
            # ファイルアップローダーはフォームの外でも良いが、クリアを考えると中に入れる
            uploaded_file = st.file_uploader("構成案の画像ファイル (JPG, PNG) を選択してください", type=["png", "jpg", "jpeg"])

            st.subheader("2. テキスト指示")
            impression_text = st.text_input("全体の雰囲気・スタイルは？ (例: ダークでスタイリッシュ)")
            details_text = st.text_area("各要素の詳細指示 (例: A:ロゴ KAWAI DESIGN, B:見出し AI駆動...) ※改行して入力してください", height=200)

            st.subheader("3. 生成サイズ")
            dalle_size = st.selectbox(
                "生成したい画像のサイズを選択 (DALL-E 3)",
                ("1024x1024", "1792x1024", "1024x1792"),
                index=0
            )

            # フォームの送信ボタン（これが従来の生成ボタンの役割）
            generate_button = st.form_submit_button("🖼️ ラフ画像を生成する", type="primary")
            # ▲▲▲ ここまでがフォーム ▲▲▲

        # アップロード画像のプレビューはフォームの外に置くことも可能
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='アップロードされた構成案 (再生成後も表示されます)', use_column_width=True)
            except Exception as e:
                st.error(f"画像プレビューエラー: {e}")

    # --- ボタンが押された後の処理 ---
    # generate_button はフォーム送信時にTrueになる
    if generate_button:
        # フォーム内で定義された uploaded_file, impression_text などにアクセスできる
        if uploaded_file is not None and details_text:
            layout_image_bytes = uploaded_file.getvalue()
            with col2: # 右カラムに結果を表示
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です...しばらくお待ちください...'):
                    # --- API連携処理 (前回と同じ) ---
                    st.info("Step 1/3: 構成案画像を解析中 (Gemini Vision)...")
                    layout_info = analyze_layout_with_gemini(layout_image_bytes, GOOGLE_API_KEY)
                    if layout_info:
                        with st.expander("レイアウト解析結果"): st.text(layout_info)
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (Gemini Text)...")
                        dalle_prompt = generate_dalle_prompt_with_gemini(layout_info, impression_text, details_text, dalle_size, GOOGLE_API_KEY)
                        if dalle_prompt:
                             with st.expander("生成されたDALL-Eプロンプト"): st.text(dalle_prompt)
                             st.info("Step 3/3: 画像を生成中 (DALL-E 3)...")
                             image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)
                             if image_url:
                                  st.success("🎉 画像生成が完了しました！")
                                  st.subheader("生成されたラフ画像")
                                  try:
                                       image_response = requests.get(image_url)
                                       image_response.raise_for_status()
                                       img_data = image_response.content
                                       st.image(img_data, caption='生成されたラフ画像', use_column_width=True)
                                       st.balloons()
                                  except Exception as download_e:
                                       st.error(f"画像ダウンロード/表示エラー: {download_e}")
                                       st.write(f"画像URL: {image_url}")
        else:
            with col1: # エラーメッセージは左カラムに表示
                 st.warning("👈 構成案の画像アップロードと、各要素の詳細指示を入力してからボタンを押してください。")

# (secrets_ok が False の場合の処理は省略、必要なら追加)
