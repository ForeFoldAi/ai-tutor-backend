import sys
import os
import json

# Ensure ai-tutor-backend is on python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.textbook_structure import detect_textbook_structure

def test_extraction(pdf_path: str):
    print(f"\n=======================================================")
    print(f"Universal Textbook Structure Engine Extraction Test")
    print(f"File: {pdf_path}")
    print(f"=======================================================\n")

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    result = detect_textbook_structure(pdf_path)

    profile = result.get("profile", {})
    chapters = result.get("chapters", [])
    non_chapters = result.get("non_chapter_sections", [])

    print("📄 1. DOCUMENT PROFILE & CALIBRATION:")
    print(f"  • Total PDF Pages       : {result.get('total_pages')}")
    print(f"  • PDF Type              : {result.get('pdf_type')}")
    print(f"  • Baseline Font Size    : {profile.get('baseline_font_size')} pt")
    print(f"  • Native PDF Bookmarks  : {'Yes' if profile.get('has_bookmarks') else 'No'}")
    print(f"  • Calibrated Page Offset: +{result.get('page_offset')} (Physical PDF Page = Printed Page + {result.get('page_offset')})")
    print(f"  • Offset Confidence     : {result.get('offset_confidence', 0.0) * 100:.1f}%")
    print(f"  • Structure Confidence  : {result.get('confidence_score', 0.0) * 100:.1f}%")
    print(f"  • Detection Method      : {result.get('detection_method')}")
    print(f"  • Needs Manual Review   : {'Yes' if result.get('needs_manual_review') else 'No'}")

    cleaner_hints = profile.get("cleaner_hints", {})
    watermarks = cleaner_hints.get("detected_watermarks", [])
    headers_footers = cleaner_hints.get("repeated_headers_footers", [])
    print(f"  • Detected Watermarks   : {watermarks if watermarks else 'None'}")
    print(f"  • Boilerplate Patterns  : {len(headers_footers)} recurring patterns identified")

    print("\n⚠️ 2. REVIEW WARNINGS & INTEGRITY CHECKS:")
    warnings = result.get("review_warnings", [])
    if warnings:
        for w in warnings:
            print(f"  • {w}")
    else:
        print("  • No warnings! Clean structure resolution with complete boundary coverage.")

    print(f"\n📑 3. NON-CHAPTER SECTIONS DETECTED ({len(non_chapters)} sections):")
    for nc in non_chapters:
        title = nc.get("chapter_title", "")
        s_pdf = nc.get("start_pdf_page", 0) + 1
        e_pdf = nc.get("end_pdf_page", 0) + 1
        stype = nc.get("section_type", "front_matter")
        print(f"  • [{stype.upper()}] \"{title}\" -> PDF pp. {s_pdf}-{e_pdf}")

    print(f"\n📚 4. EXTRACTED CHAPTERS / UNITS ({len(chapters)} main chapters):")
    print(f"{'-'*115}")
    print(f"{'#':<6} | {'Level':<8} | {'Chapter / Unit Title':<45} | {'PDF Range':<15} | {'Printed Range':<14} | {'Conf':<6}")
    print(f"{'-'*115}")

    for chap in chapters:
        num = chap.get("chapter_number", "")
        level = chap.get("hierarchy_level", "UNIT").upper()
        title = chap.get("chapter_title", "")
        disp_title = (title[:42] + '...') if len(title) > 45 else title
        start_pdf = chap.get("start_pdf_page", 0) + 1  # 1-indexed for human display
        end_pdf = chap.get("end_pdf_page", 0) + 1
        pdf_range = f"pp. {start_pdf}–{end_pdf}"
        
        pr_start = chap.get("printed_start_page")
        pr_end = chap.get("printed_end_page")
        pr_range = f"pp. {pr_start}–{pr_end}" if pr_start and pr_end else (f"p. {pr_start}" if pr_start else "N/A")
        conf = f"{chap.get('confidence_score', 0.0) * 100:.0f}%"

        print(f"{num:<6} | {level:<8} | {disp_title:<45} | {pdf_range:<15} | {pr_range:<14} | {conf:<6}")

        # Sub-readings / sub-chapters
        sub_chaps = chap.get("sub_chapters", [])
        for sub in sub_chaps:
            sub_num = sub.get("chapter_number", "")
            sub_title = sub.get("chapter_title", "")
            sub_disp = ("  ↳ " + sub_title)[:45]
            sub_level = sub.get("hierarchy_level", "READING").upper()
            s_pdf = sub.get("start_pdf_page", 0) + 1
            e_pdf = sub.get("end_pdf_page", 0) + 1
            s_pdf_range = f"pp. {s_pdf}–{e_pdf}"
            s_pr_start = sub.get("printed_start_page")
            s_pr_end = sub.get("printed_end_page")
            s_pr_range = f"pp. {s_pr_start}–{s_pr_end}" if s_pr_start and s_pr_end else (f"p. {s_pr_start}" if s_pr_start else "N/A")
            s_conf = f"{sub.get('confidence_score', 0.0) * 100:.0f}%"
            print(f"{sub_num:<6} | {sub_level:<8} | {sub_disp:<45} | {s_pdf_range:<15} | {s_pr_range:<14} | {s_conf:<6}")

    print(f"{'-'*115}\n")

if __name__ == "__main__":
    pdf_target = sys.argv[1] if len(sys.argv) > 1 else "/Applications/ForeFold/virtual tutor/8th eng.pdf"
    test_extraction(pdf_target)
