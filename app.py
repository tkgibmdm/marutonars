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
# (GitHubリポジトリのルートに 'prompts' フォルダがあり、その中にファイルがあると想定)
layout_analysis_prompt_text = load_prompt("prompts/layout_analysis_prompt.txt")
dalle_instruction_template_text = load_prompt("prompts/dalle_prompt_instruction_template.txt")
prompts_loaded = layout_analysis_prompt_text is not None and dalle_instruction_template_text is not None

# --- Load API Keys ---
secrets_ok = False
if prompts_loaded: # プロンプトが正常に読み込めたら次に進む
    try:
        # GOOGLE_API_KEY は現在使わないが、Secrets設定はそのまま読み込み確認
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # こちらを主に使用
        if not GOOGLE_API_KEY or not OPENAI_API_KEY:
            st.error("エラー: Streamlit CloudのSecretsに必要なAPIキーが設定されていないか、値が空です。(GOOGLE_API_KEY, OPENAI_API_KEY)")
            st.stop()
        secrets_ok = True
    except KeyError as e:
        st.error(f"エラー: Streamlit CloudのSecretsにキー '{e}' が見つかりません。名前が正しいか確認してください。")
        st.stop()
    except Exception as e:
         st.error(f"Secrets読み込み中に予期せぬエラーが発生しました: {e}")
         st.stop()

# --- API Function Definitions ---
# (APIキーとプロンプトが正常に読み込めた場合のみ定義・実行)
if secrets_ok and prompts_loaded:

    # ▼▼▼ GPT-4o (Vision) でレイアウト解析 ▼▼▼
    # 引数に layout_prompt_text を追加
    def analyze_layout_with_gpt4o(image_bytes, api_key, layout_prompt_text):
        """OpenAI GPT-4o APIを呼び出し、レイアウト情報を抽出する関数"""
        if not layout_prompt_text:
             st.error("レイアウト解析プロンプトが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            try:
                # MIMEタイプをより正確に判定する試み
                img = Image.open(BytesIO(image_bytes))
                img_format = img.format
                if img_format == 'PNG': mime_type = "image/png"
                elif img_format in ['JPEG', 'JPG']: mime_type = "image/jpeg"
                # 他のフォーマットも必要に応じて追加 (GIFなど)
                else: mime_type = "image/jpeg" # デフォルト
            except Exception:
                mime_type = "image/jpeg" # 判定失敗時のデフォルト

            prompt_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": layout_prompt_text}, # ファイルから読み込んだプロンプトを使用
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]}
            ]
            response = client.chat.completions.create(model="gpt-4o", messages=prompt_messages, max_tokens=1000)
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.warning(f"GPT-4o応答なし（レイアウト解析）. Response: {response}")
                return None
        except Exception as e:
            st.error(f"OpenAI GPT-4o APIエラー (レイアウト解析): {e}")
            return None
    # ▲▲▲ GPT-4o (Vision) でレイアウト解析 ▲▲▲

    # ▼▼▼ GPT-4o (Text) でDALL-Eプロンプト生成 ▼▼▼
    # 引数に dalle_instruction_template_text を追加
    def generate_dalle_prompt_with_gpt4o(layout_info, impression, details, size, api_key, dalle_instruction_template_text):
        """OpenAI GPT-4o APIを呼び出し、DALL-E 3用プロンプトを生成する関数 (テキスト直接描画試行)"""
        if not dalle_instruction_template_text:
             st.error("DALL-Eプロンプト生成指示テンプレートが読み込めていません。")
             return None
        try:
            client = OpenAI(api_key=api_key)
            # --- テンプレートに動的な値を埋め込む ---
            # .txtファイルの内容（テンプレート）に .format() を適用
            prompt_generation_instruction = dalle_instruction_template_text.format(
                size=size, layout_info=layout_info, impression=impression, details=details
            )
            # --- 埋め込みここまで ---
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
    # ▲▲▲ GPT-4o (Text) でDALL-Eプロンプト生成 ▲▲▲

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

    # ▼▼▼ テキスト描画関数 (Task 2.1で追加したもの) ▼▼▼
    def add_text_to_image(image, text, position, font_path, font_size, text_color=(0, 0, 0, 255)):
        """Pillowを使って画像にテキストを描画する関数"""
        try:
            base = image.convert("RGBA")
            txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                st.error(f"エラー: 指定されたフォントファイルが見つからないか、読み込めません: {font_path}")
                return None # フォント読み込み失敗
            draw = ImageDraw.Draw(txt_layer)
            draw.text(position, text, font=font, fill=text_color)
            out = Image.alpha_composite(base, txt_layer)
            return out.convert("RGB") # RGBで返す
        except Exception as e:
            st.error(f"テキスト描画中にエラーが発生しました: {e}")
            return None
    # ▲▲▲ テキスト描画関数 ▲▲▲

    # --- Streamlit App Main UI ---
    st.title("🤖 AIバナーラフ生成プロトタイプ (GPT-4o Ver.)")
    st.write("構成案の画像とテキスト指示から、AIがバナーラフ画像を生成します。")

    col1, col2 = st.columns(2)
    with col1:
        # フォーム (clear_on_submit はデバッグのため一旦外したままにします)
        with st.form("input_form"): # clear_on_submit=True は必要に応じて後で戻してください
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
                 # Read image for preview display
                 image_preview = Image.open(uploaded_file)
                 st.image(image_preview, caption='アップロードされた構成案', use_column_width=True)
             except Exception as e:
                 # エラーが出てもプレビューは止めない
                 # st.error(f"画像プレビューエラー: {e}")
                 pass


    # --- ボタンが押された後の処理 (Corrected Calls) ---
    if generate_button:
        # Check inputs again
        if uploaded_file is not None and details_text:
            # Get file bytes immediately after button press
            layout_image_bytes = uploaded_file.getvalue()
            with col2:
                st.subheader("⚙️ 生成プロセス")
                with st.spinner('AIが画像を生成中です... (GPT-4o x2 + DALL-E 3)'):

                    # --- Step 1: レイアウト解析 (修正済み呼び出し) ---
                    st.info("Step 1/3: 構成案画像を解析中 (GPT-4o Vision)...")
                    # ★★★ layout_analysis_prompt_text を引数として渡す ★★★
                    layout_info = analyze_layout_with_gpt4o(layout_image_bytes, OPENAI_API_KEY, layout_analysis_prompt_text)

                    if layout_info:
                        with st.expander("レイアウト解析結果 (GPT-4o)", expanded=False): st.text(layout_info)

                        # --- Step 2: DALL-E プロンプト生成 (修正済み呼び出し) ---
                        st.info("Step 2/3: DALL-E 3用プロンプトを生成中 (GPT-4o Text)...")
                        # ★★★ dalle_instruction_template_text を引数として渡す ★★★
                        dalle_prompt = generate_dalle_prompt_with_gpt4o(layout_info, impression_text, details_text, dalle_size, OPENAI_API_KEY, dalle_instruction_template_text)

                        if dalle_prompt:
                             # Check if dalle_prompt indicates failure
                             if "sorry" in dalle_prompt.lower():
                                  st.error("GPT-4oがプロンプト生成に必要な情報を得られなかったか、処理を拒否したようです。")
                             with st.expander("生成されたDALL-Eプロンプト (GPT-4o)", expanded=True): st.text(dalle_prompt)

                             # Only proceed if prompt generation seems successful
                             if "sorry" not in dalle_prompt.lower():
                                 # --- Step 3: 画像生成 (変更なし) ---
                                 st.info("Step 3/3: 画像を生成中 (DALL-E 3)...")
                                 image_url = generate_image_with_dalle3(dalle_prompt, dalle_size, OPENAI_API_KEY)

                                 if image_url:
                                      st.success("🎉 画像生成が完了しました！")
                                      st.subheader("生成されたラフ画像")
                                      # --- Step 4: 画像表示 ---
                                      try:
                                           image_response = requests.get(image_url); image_response.raise_for_status()
                                           img_data = image_response.content
                                           # ★★★ ここでテキスト描画関数を呼び出す処理を追加する（Task 2.1以降）★★★
                                           # 現時点では DALL-E 3 の画像をそのまま表示
                                           st.image(img_data, caption='生成されたラフ画像 (テキスト未描画)', use_column_width=True)
                                           st.balloons()
                                      except Exception as download_e:
                                           st.error(f"画像ダウンロード/表示エラー: {download_e}"); st.write(f"画像URL: {image_url}")
        else:
            # Show warning inside the main column if inputs are missing on submit
             with col1:
                st.warning("👈 画像のアップロードと詳細指示を入力してからボタンを押してください。")

# --- Error handling for failed prompt/secret loading ---
elif not prompts_loaded:
     st.error("プロンプトファイルの読み込みに失敗したため、アプリを起動できません。GitHubリポジトリ内の 'prompts' フォルダとファイルを確認してください。")
elif not secrets_ok:
     st.warning("アプリの初期化中にSecrets関連で問題が発生したため、UIを表示できません。ログやSecrets設定を確認してください。")

```

**主な変更点:**

* ボタン処理の中で `analyze_layout_with_gpt4o` と `generate_dalle_prompt_with_gpt4o` を呼び出す際に、ファイルから読み込んだプロンプト/テンプレート変数を正しく引数として渡すように修正しました。
* Task 2.1 で定義した `add_text_to_image` 関数もコード内に含めましたが、**まだ呼び出す処理は入れていません**（コメントで「★★★ ここでテキスト描画関数を呼び出す処理を追加する（Task 2.1以降）★★★」と記載）。現在の動作は DALL-E 3 が生成した画像をそのまま表示します。

**次のステップ:**

1.  このコードで GitHub の `app.py` を**上書き**してください。
2.  変更をコミット＆プッシュします。
3.  Streamlit Community Cloud でアプリが再デプロイされるのを待ちます（またはReboot）。
4.  アプリで画像生成を実行し、今度こそ `TypeError` が発生せずに最後まで処理が実行されるか確認してください。

これでエラーが解消されるはずです。ご確認くだ
