import streamlit as st

from app.rag.fact_checker import fact_check_claim


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="FactCheck RAG",
    page_icon="🔎",
    layout="wide",
)

st.title("FactCheck RAG")
st.caption(
    "Evidence-grounded claim verification with four transparent audit metrics."
)

st.markdown(
    """
    Enter a factual claim to retrieve supporting evidence and evaluate it.
    The result includes the verdict, explanation, four audit metrics,
    and the retrieved evidence used for verification.
    """
)


# --------------------------------------------------
# Display helpers
# --------------------------------------------------

def normalize_list(value):
    """Normalize audit fields into lists for consistent UI rendering."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    # Avoid accidentally treating a string as an iterable of characters.
    return [value]


def display_list(items, empty_message):
    """Display strings and structured audit items."""

    items = normalize_list(items)

    if not items:
        st.info(empty_message)
        return

    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            with st.container(border=True):
                for key, value in item.items():
                    label = key.replace("_", " ").capitalize()

                    st.markdown(f"**{label}:**")

                    if isinstance(value, list):
                        if value:
                            for value_item in value:
                                st.write(f"- {value_item}")
                        else:
                            st.write("None")
                    elif isinstance(value, dict):
                        st.json(value)
                    else:
                        st.write(value)
        else:
            st.markdown(f"**{index}.** {item}")


def display_evidence(evidence):
    """Display local and web evidence with source URLs."""

    if not evidence:
        st.info("No evidence was retrieved.")
        return

    for item in evidence:
        evidence_id = item.get("evidence_id", "Unknown")
        source = item.get("source", "Unknown")
        source_type = item.get("source_type", "unknown")
        content = item.get("content", "")
        url = item.get("url")

        with st.expander(f"{evidence_id} | {source}"):
            st.caption(f"Evidence type: {source_type}")

            if item.get("chunk_id") is not None:
                st.caption(f"Chunk ID: {item['chunk_id']}")

            st.write(content)

            if url:
                st.markdown(f"[Open source]({url})")


# --------------------------------------------------
# Fact-check result
# --------------------------------------------------

def display_fact_check_result(result):
    """Display verdict and all four required audit metrics."""

    if not isinstance(result, dict):
        st.error("The fact-checking pipeline did not return a valid result.")
        return

    st.divider()
    st.header("Fact-check result")

    st.markdown("**Claim submitted**")
    st.write(result.get("claim", "Not provided"))

    # Verdict
    verdict = result.get("verdict", "Insufficient Evidence")
    explanation = result.get(
        "explanation",
        "No explanation was returned.",
    )

    if verdict == "Fully True":
        st.success(f"Verdict: {verdict}")
    elif verdict == "Partially True":
        st.warning(f"Verdict: {verdict}")
    elif verdict == "False":
        st.error(f"Verdict: {verdict}")
    else:
        st.info(f"Verdict: {verdict}")

    st.markdown("**Explanation**")
    st.write(explanation)

    # Retrieve all four required metrics.
    assumptions = normalize_list(
        result.get("assumptions_made", [])
    )
    external_info = normalize_list(
        result.get("external_information_added", [])
    )
    verifiable = normalize_list(
        result.get("verifiable_portions", [])
    )
    unverifiable = normalize_list(
        result.get("unverifiable_portions", [])
    )

    # --------------------------------------------------
    # Four-metric overview
    # --------------------------------------------------

    st.divider()
    st.header("Transparency and Audit Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="1. Assumptions Made",
            value=len(assumptions),
        )

        st.metric(
            label="2. External Information Added",
            value=len(external_info),
        )

    with col2:
        st.metric(
            label="3. Portions That Can Be Verified",
            value=len(verifiable),
        )

        st.metric(
            label="4. Portions That Cannot Be Verified",
            value=len(unverifiable),
        )
        web_research = result.get(
    "external_web_research_performed",
    False,
)
    web_results = result.get("web_search_results_found", 0)

    st.caption(
        f"External web search attempted: {'Yes' if web_research else 'No'}"
    )
    st.caption(f"External web evidence retrieved: {web_results}")

    if result.get("web_search_error"):
        st.warning(
            "Web fallback was attempted but failed. "
            "The verdict may rely only on local evidence."
        )

    # --------------------------------------------------
    # Detailed audit sections
    # These remain visible without needing to expand panels.
    # --------------------------------------------------

    st.subheader("1. List of Assumptions Made")
    display_list(
        assumptions,
        "No assumptions were identified by the analysis.",
    )

    st.subheader("2. External Information Added")
    display_list(
        external_info,
        "No additional factual information was identified.",
    )

    web_research = result.get(
        "external_web_research_performed",
        False,
    )

    st.caption(
        "External web research performed: "
        + ("Yes" if web_research else "No")
    )

    st.subheader("3. Portions That Can Be Verified")
    display_list(
        verifiable,
        "No portions were identified as verifiable from the retrieved evidence.",
    )

    st.subheader("4. Portions That Cannot Be Verified")
    display_list(
        unverifiable,
        "No portions were identified as unverifiable.",
    )

    # --------------------------------------------------
    # Evidence and debugging
    # --------------------------------------------------

    st.divider()

    with st.expander("Retrieved Evidence and Sources"):
        display_evidence(result.get("evidence", []))

    with st.expander("Raw Response JSON"):
        st.json(result)


# --------------------------------------------------
# Claim input
# --------------------------------------------------

with st.form("fact_check_form"):
    claim = st.text_area(
        "Enter a claim to fact-check",
        placeholder="Example: Australia won the 2003 Cricket World Cup.",
        height=120,
        max_chars=5000,
    )

    submitted = st.form_submit_button(
        "Fact-check claim",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Run fact-checking pipeline
# --------------------------------------------------

if submitted:
    if not claim or not claim.strip():
        st.warning("Please enter a claim before submitting.")

    else:
        try:
            with st.spinner(
                "Retrieving evidence and analyzing the claim..."
            ):
                result = fact_check_claim(claim.strip())

            display_fact_check_result(result)

        except ValueError as exc:
            st.error(f"Invalid input or response: {exc}")

        except Exception as exc:
            st.error(
                "The fact-checking pipeline encountered an error."
            )

            with st.expander("Technical error details"):
                st.exception(exc)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "FactCheck RAG | Verdicts depend on the available evidence. "
    "Insufficient evidence does not automatically mean a claim is false."
)