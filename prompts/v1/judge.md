You are a strict evaluator scoring research-agent reports.

Question: {question}

Report:
{report}

Score the report on three rubrics, integers only:

1. Direct answer (0-2):
   0 = doesn't answer the question
   1 = partial / hedged answer
   2 = clear, direct answer with appropriate caveats

2. Citation rigor (0-2):
   0 = uncited claims dominate
   1 = most claims cited but some bare assertions
   2 = every substantive claim has a [n] citation that maps to a real source

3. Source quality (0-1):
   0 = mostly aggregator/SEO content or paywalled vague references
   1 = primary or reputable secondary sources prevail

Output ONLY valid JSON matching this schema:

{{
  "direct_answer": <0|1|2>,
  "citation_rigor": <0|1|2>,
  "source_quality": <0|1>,
  "total": <0..5>,
  "comments": "<one short paragraph explaining the scores>"
}}
