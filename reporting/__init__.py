"""
reporting package - Milestone 4.
Modules:
    pdf_report.py - M4.2: professional PDF export built from stored
                    evaluation results.
The document is first assembled as a plain Python structure, so its exact
contents can be asserted against the underlying evaluation records without
opening a PDF, and only then rendered to PDF bytes.
"""
from reporting import pdf_report
