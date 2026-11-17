# FSSAI source manifest — refresh before any real compliance use

**Last source review:** 16 November 2026 (India).

The `.txt` files shipped beside this manifest are explicitly **sample/demo extracts**. They are not the authoritative corpus and must not be used to make real compliance-clearance claims. The production decision engine now blocks positive clearance while no authoritative source chunk is retrieved.

## Official FSSAI source pages to download and version

1. **Labelling and Display Regulations, 2020 — amendments index:** https://www.fssai.gov.in/food-law/regulations/amendments/labelling-display
   - FSSAI lists amendments dated 10 September 2021, 9 September 2022, 11 October 2022, 8 August 2025, 24 March 2026 and 6 November 2026.
   - The 24 March 2026 First Amendment is stated in the Gazette notice to commence on 1 July 2027.
   - The 6 November 2026 Second Amendment concerning the graphic specifications of the milk logo is stated to commence on 1 July 2028.
2. **Advertising and Claims Regulations, 2018 — amendments index:** https://www.fssai.gov.in/food-law/regulations/amendments/advertising-claims
   - FSSAI lists amendments dated 9 October 2020, 30 August 2022 and 13 December 2022.
3. **Packaging Regulations, 2018 — amendments index:** https://www.fssai.gov.in/food-law/regulations/amendments/packaging
   - FSSAI lists amendments dated 27 January 2022, 30 August 2022, 28 March 2025 and 7 August 2026.
4. **FSSAI notifications index:** https://www.fssai.gov.in/food-law/notifications
   - Check this index for later notifications, corrigenda, implementation directions and effective dates.
5. **FSSAI all regulations index:** https://www.fssai.gov.in/food-law/regulations
   - Check applicable product standards, additives, contaminants, packaging, licensing and category-specific regulations in addition to label rules.

## Ingestion requirements

- Download the full consolidated/compendium text and every subsequent Gazette amendment for each applicable regulation.
- Preserve the original PDF and record its official title, notification number, Gazette date, effective date, retrieval date, and FSSAI URL.
- Do not mark a draft notification as binding law.
- Do not apply a future-effective amendment before its commencement date.
- Re-index the complete, reconciled corpus as one versioned release; do not blend old and new clauses without effective-date metadata. Keep future-effective amendments stored but inactive until their stated commencement date.
- Have a qualified regulatory specialist validate the corpus and rule mappings before relying on results.

**Operational limitation:** the manifest records source discovery and amendment chronology only. It does not mean these documents are already downloaded, reconciled or indexed in this repository.

## Explicit source authorization

The ingestion pipeline treats every PDF as sample evidence unless the filename and complete-file SHA-256 match an explicit `REGULATORY_AUTHORIZED_SHA256_JSON` environment allowlist. Example shape: `{"FSSAI_Labelling_Display_Compendium.pdf":"<64-hex-sha256>"}`. Only add a digest after a person has obtained the PDF from an official source, verified the Gazette/FSSAI publication details and effective dates, and had the corpus reviewed. This checksum is a deployment attestation, not a cryptographic signature supplied by FSSAI. A new document version requires a new review and digest. All `.txt` documents remain sample-only.
