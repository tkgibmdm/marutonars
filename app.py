# 修正前 (Colab用)
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# OPENAI_API_KEY = userdata.get('marutonars')

# 修正後 (Streamlit Community Cloud用)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not GOOGLE_API_KEY or not OPENAI_API_KEY:
        st.error("エラー: Streamlit CloudのSecretsにAPIキーが設定されていません。")
        st.stop()
except KeyError as e:
    st.error(f"エラー: Streamlit CloudのSecretsに {e} が見つかりません。")
    st.stop()
except Exception as e:
     st.error(f"Secrets読み込みエラー: {e}")
     st.stop()

# 不要になる可能性のある行を削除 (もしあれば)
# from google.colab import userdata
