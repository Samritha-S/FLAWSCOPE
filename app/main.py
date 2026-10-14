"""Streamlit web dashboard entry point for FlawScope."""

import streamlit as st


def main():
    st.set_page_config(page_title="FlawScope", layout="wide")
    st.title("FlawScope")
    st.write("Counterpointing speech analytics and temporal flaw grounding.")


if __name__ == "__main__":
    main()
