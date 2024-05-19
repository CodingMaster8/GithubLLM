import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Home",
        page_icon="👋",
    )

    st.write("# Welcome to the time machine demo 👋")

    st.write("""
            Please go to the sidebar and:
             - Load some data
             - Use the time machine             
             """)



if __name__ == "__main__":
    run()

