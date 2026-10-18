import streamlit as st

from app.rag.fact_checker import fact_check_claim

def display_fact_check_result(result: dict) -> None:
    """Display the verdict and all four required audit metrics."""

    st.subheader("Fact-check result")

    st.write("**Claim:**", result["claim"])
    st.write("**Verdict:**", result["verdict"])
    st.write("**Explanation:**", result["explanation"])

    st.divider()
    st.subheader("Transparency and audit metrics")

    # Metric 1: Assumptions made
    with st.expander("1. Assumptions Made", expanded=True):
        items = result["assumptions_made"]

        if items:
            for item in items:
                if isinstance(item, dict):
                    st.markdown(f"**Assumption:** {item.get('assumption', '')}")
                    st.write(item.get("reason", ""))
                else:
                    st.write(item)
        else:
            st.info("No assumptions identified.")

    # Metric 2: External information added
    with st.expander("2. External Information Added", expanded=True):
        items = result["external_information_added"]

        if items:
            for item in items:
                if isinstance(item, dict):
                    st.markdown(f"**Added detail:** {item.get('statement', '')}")
                    st.write("Source:", item.get("source", "Unknown"))
                    st.write("Source type:", item.get("source_type", "Unknown"))
                    st.write("Evidence IDs:", item.get("evidence_ids", []))
                else:
                    st.write(item)
        else:
            st.info("No additional factual details identified.")

        st.caption(
            "External web research performed: "
            + str(result.get("external_web_research_performed", False))
        )

    # Metric 3: Portions that can be verified
    with st.expander("3. Portions That Can Be Verified", expanded=True):
        items = result["verifiable_portions"]

        if items:
            for item in items:
                if isinstance(item, dict):
                    st.markdown(f"**Statement:** {item.get('statement', '')}")
                    st.write("Status:", item.get("status", "Unknown"))
                    st.write("Reason:", item.get("reason", ""))
                    st.write("Evidence IDs:", item.get("evidence_ids", []))
                else:
                    st.write(item)
        else:
            st.info("No portions could be verified from the available evidence.")

    # Metric 4: Portions that cannot be verified
    with st.expander("4. Portions That Cannot Be Verified", expanded=True):
        items = result["unverifiable_portions"]

        if items:
            for item in items:
                if isinstance(item, dict):
                    st.markdown(f"**Statement:** {item.get('statement', '')}")
                    st.write("Reason:", item.get("reason", ""))
                    st.write("Evidence IDs:", item.get("evidence_ids", []))
                else:
                    st.write(item)
        else:
            st.success("No unverifiable portions identified.")

    # Show the retrieved evidence supporting the analysis.
    with st.expander("Retrieved Evidence", expanded=False):
        if result["evidence"]:
            for item in result["evidence"]:
                st.markdown(
                    f"**[{item['evidence_id']}] {item['source']}**"
                )
                st.caption(f"Chunk ID: {item['chunk_id']}")
                st.write(item["content"])
                st.divider()
        else:
            st.info("No evidence was retrieved.")
