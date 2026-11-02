"""Login / sign-up gate shared by every Streamlit page."""

import streamlit as st

from src import api_client
from src.theme import hero


def require_login() -> None:
    """Show the login screen and stop the page until the user signs in."""
    if st.session_state.get("token"):
        with st.sidebar:
            st.markdown(f"👤 **{st.session_state['username']}**")
            if st.button("Log out", key="logout_button"):
                st.session_state.clear()
                st.rerun()
        return

    # Hide the page navigation until the user is authenticated.
    st.markdown(
        '<style>[data-testid="stSidebar"] {display: none;}</style>',
        unsafe_allow_html=True,
    )
    hero("🩺 MediShield", "Sign in to continue.")

    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        login_tab, signup_tab = st.tabs(["Log in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Log in", type="primary", width="stretch"
                )
            if submitted:
                _authenticate(api_client.login, username, password)

        with signup_tab:
            with st.form("signup_form"):
                new_user = st.text_input(
                    "Username",
                    help="3-40 characters: letters, numbers, . _ -",
                )
                new_pass = st.text_input(
                    "Password", type="password",
                    help="At least 8 characters.",
                )
                confirm = st.text_input("Confirm password", type="password")
                created = st.form_submit_button(
                    "Create account", type="primary", width="stretch"
                )
            if created:
                if new_pass != confirm:
                    st.error("Passwords do not match.")
                else:
                    _authenticate(api_client.signup, new_user, new_pass)

    st.stop()


def _authenticate(action, username: str, password: str) -> None:
    try:
        with st.spinner("Contacting server..."):
            result = action(username, password)
    except api_client.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state["token"] = result["token"]
    st.session_state["username"] = result["username"]
    st.rerun()
