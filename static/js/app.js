const form = document.querySelector("#analysis-form");
const sourceText = document.querySelector("#source-text");
const suspiciousText = document.querySelector("#suspicious-text");
const reportPanel = document.querySelector("#report-panel");
const statusMessage = document.querySelector("#status-message");
const loadSampleButton = document.querySelector("#load-sample");
const sampleSelect = document.querySelector("#sample-select");
const reportTemplate = document.querySelector("#report-template");

const samples = [
  {
    title: "01 Exact academic copying",
    source: "Plagiarism detection is an important task in academic environments because copied work can damage originality and trust. Traditional lexical methods compare word overlap and are useful when text is copied directly.",
    suspicious: "Plagiarism detection is an important task in academic environments because copied work can damage originality and trust. Traditional lexical methods compare word overlap and are useful when text is copied directly."
  },
  {
    title: "02 Strong paraphrase",
    source: "A hybrid plagiarism detector combines lexical similarity and semantic similarity so that it can detect both direct copying and meaning-preserving paraphrasing.",
    suspicious: "A combined plagiarism system uses word overlap and meaning-based comparison to find copied passages as well as rewritten text that keeps the same idea."
  },
  {
    title: "03 Low similarity",
    source: "Machine learning models can classify text by learning patterns from labeled training examples and evaluating performance on unseen test data.",
    suspicious: "Urban transportation planning focuses on road networks, public transit routes, parking policies, and pedestrian safety in crowded cities."
  },
  {
    title: "04 Topic similarity only",
    source: "Artificial intelligence can support medical diagnosis by analyzing clinical data, imaging records, and patient history.",
    suspicious: "Artificial intelligence is also used in hospitals for administrative scheduling, insurance processing, and resource planning."
  },
  {
    title: "05 Mixed copied and original",
    source: "Sentence-BERT produces dense sentence embeddings that make semantic comparison easier. These embeddings allow similar meanings to be compared even when the wording changes.",
    suspicious: "Sentence-BERT produces dense sentence embeddings that make semantic comparison easier. In our project, the web interface also allows users to generate a report from pasted text."
  },
  {
    title: "06 Research abstract paraphrase",
    source: "The proposed framework integrates exact matching, TF-IDF similarity, semantic similarity, and document-level aggregation to improve plagiarism detection.",
    suspicious: "The system combines exact overlap, TF-IDF based lexical scoring, semantic scoring, and final document aggregation to make plagiarism detection stronger."
  },
  {
    title: "07 Software engineering unrelated",
    source: "Version control systems help teams track code changes, review contributions, and recover previous versions of a project.",
    suspicious: "Photosynthesis allows green plants to convert sunlight into chemical energy using chlorophyll, carbon dioxide, and water."
  },
  {
    title: "08 Near copy with synonyms",
    source: "Threshold tuning is important because different models produce scores on different scales and should not be evaluated with one fixed cutoff.",
    suspicious: "Threshold calibration is necessary because separate models generate values on different ranges and should not be judged using the same cutoff."
  },
  {
    title: "09 Same meaning different structure",
    source: "The hybrid model improved precision by avoiding decisions based on a single high-scoring sentence pair.",
    suspicious: "Precision increased because the final system no longer allowed one unusually similar sentence to control the whole document decision."
  },
  {
    title: "10 Exact technical sentence",
    source: "The final hybrid score combines local sentence evidence with global TF-IDF and global SBERT similarity.",
    suspicious: "The final hybrid score combines local sentence evidence with global TF-IDF and global SBERT similarity."
  },
  {
    title: "11 Partial plagiarism",
    source: "Bootstrap confidence intervals help estimate how stable the reported metrics are when the test set is small.",
    suspicious: "Bootstrap confidence intervals help estimate how stable the reported metrics are when the test set is small. The dashboard design uses clean spacing and readable cards."
  },
  {
    title: "12 Conceptual overlap",
    source: "Ablation studies remove one component at a time to measure how much each part contributes to the final performance.",
    suspicious: "Component analysis can be performed by taking away one signal at a time and observing the change in the final model score."
  },
  {
    title: "13 Non-plagiarized same domain",
    source: "TF-IDF assigns importance to words by considering how often they appear in a document and how rare they are across a collection.",
    suspicious: "Word embeddings represent tokens as vectors in a continuous space so that neural models can learn relationships between words."
  },
  {
    title: "14 Student assignment paraphrase",
    source: "The internet has made information easier to access, but it has also increased the risk of copying content without proper citation.",
    suspicious: "Online resources are easy to reach, yet this convenience has also made uncited copying more common among students."
  },
  {
    title: "15 High semantic low lexical",
    source: "The evaluation protocol used a fixed random seed so that the experiment could be reproduced consistently.",
    suspicious: "To make the experiment repeatable, the data split was controlled by setting the same random value each time."
  },
  {
    title: "16 Low risk different field",
    source: "Cloud computing provides scalable infrastructure by allowing users to access computing resources over the internet.",
    suspicious: "Classical music theory studies harmony, rhythm, melody, and the structure of musical compositions."
  },
  {
    title: "17 Methodology copy",
    source: "Candidate sentence pairs are filtered using both lexical and semantic evidence before they are used in document-level aggregation.",
    suspicious: "Candidate sentence pairs are filtered using both lexical and semantic evidence before they are used in document-level aggregation."
  },
  {
    title: "18 Report wording paraphrase",
    source: "The results should be interpreted cautiously because the held-out test set contains only forty-nine document pairs.",
    suspicious: "The findings should not be overstated since the final test split includes just forty-nine pairs of documents."
  },
  {
    title: "19 Literature review overlap",
    source: "Recent plagiarism detection research increasingly combines lexical features with transformer-based semantic representations.",
    suspicious: "Modern work on plagiarism detection often mixes word-level features with transformer embeddings to improve detection."
  },
  {
    title: "20 Noisy copied fragment",
    source: "The PAN metadata file was used to construct positive pairs, while negative pairs were sampled from non-linked document combinations.",
    suspicious: "Our system also includes a visual interface. The PAN metadata file was used to construct positive pairs, while negative pairs were sampled from non-linked document combinations."
  },
  {
    title: "21 Interview explanation paraphrase",
    source: "TF-IDF is strong for copied text, SBERT is useful for paraphrased meaning, and the hybrid model combines both signals.",
    suspicious: "The lexical method works well when text is copied, the semantic model helps with rewritten meaning, and the combined model uses the strengths of both."
  },
  {
    title: "22 Independent writing",
    source: "Data preprocessing removes noise from raw text and converts it into a cleaner form for machine learning models.",
    suspicious: "Renewable energy systems reduce dependence on fossil fuels by using sources such as solar, wind, hydro, and biomass power."
  },
  {
    title: "23 Academic conclusion rewrite",
    source: "The proposed system provides a practical basis for future plagiarism-screening tools, although broader validation is still required.",
    suspicious: "This framework can support future plagiarism checking systems, but it still needs to be tested on larger and more realistic datasets."
  },
  {
    title: "24 Short exact reuse",
    source: "Precision improved dramatically while recall remained high.",
    suspicious: "Precision improved dramatically while recall remained high."
  },
  {
    title: "25 Long mixed case",
    source: "The project began as a hybrid plagiarism detector using exact matching, TF-IDF, and SBERT. Later updates improved dataset construction, threshold calibration, and document-level aggregation. These changes made the final system more reliable for research presentation.",
    suspicious: "The system started as a plagiarism detector that used exact matching, TF-IDF, and SBERT. Later, the work improved how pairs were created, how thresholds were selected, and how document scores were combined. These improvements made the final project stronger for research discussion."
  }
];

function formatScore(value) {
  return Number(value || 0).toFixed(4);
}

function formatPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function evidenceBand(value) {
  const score = Number(value || 0);
  if (score >= 0.75) return "Very strong";
  if (score >= 0.50) return "Strong";
  if (score >= 0.25) return "Moderate";
  if (score >= 0.09) return "Low but suspicious";
  return "Low";
}

function explainRisk(data) {
  const score = Number(data.scores.hybrid || 0);
  const matches = Number(data.features.match_count || 0);
  const coverage = formatPercent(data.features.coverage);

  if (score >= 0.50) {
    return `High risk because the text shows strong similarity and ${matches} matched sentence pair(s).`;
  }

  if (score >= data.thresholds.hybrid) {
    return `Needs review because it crosses the suspicious threshold, with ${coverage} matched coverage.`;
  }

  return `Low risk because it stays below the suspicious threshold, with ${coverage} matched coverage.`;
}

function predictionText(flag, threshold) {
  return flag
    ? `Suspicious because score is above ${formatPercent(threshold)}`
    : `Less suspicious because score is below ${formatPercent(threshold)}`;
}

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.style.color = isError ? "#b64d57" : "#62707d";
}

function makeKeyValueList(target, rows) {
  target.innerHTML = "";
  rows.forEach(([key, value]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "kv-row";

    const dt = document.createElement("dt");
    dt.textContent = key;

    const dd = document.createElement("dd");
    dd.textContent = value;

    wrapper.append(dt, dd);
    target.append(wrapper);
  });
}

function renderMatches(target, matches) {
  target.innerHTML = "";

  if (!matches.length) {
    const empty = document.createElement("p");
    empty.textContent = "No strong sentence-level matches were found.";
    target.append(empty);
    return;
  }

  matches.forEach((match, index) => {
    const item = document.createElement("article");
    item.className = "match-item";
    item.dataset.filter = String(match.passed_filter);

    const score = document.createElement("div");
    score.className = "match-score";
    score.innerHTML = `<span>Match ${index + 1}</span><span>${formatPercent(match.hybrid_score)} match strength</span>`;

    const text = document.createElement("div");
    text.className = "match-text";
    text.innerHTML = `
      <div><strong>Source:</strong> ${escapeHtml(match.source_sentence)}</div>
      <div><strong>Suspicious:</strong> ${escapeHtml(match.suspicious_sentence)}</div>
      <div><strong>Text overlap:</strong> ${evidenceBand(match.tfidf_score)} &nbsp; <strong>Meaning similarity:</strong> ${evidenceBand(match.sbert_score)} &nbsp; <strong>Evidence quality:</strong> ${match.passed_filter ? "Useful" : "Weak"}</div>
    `;

    item.append(score, text);
    target.append(item);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderReport(data) {
  const fragment = reportTemplate.content.cloneNode(true);

  fragment.querySelector("[data-field='verdict']").textContent = data.verdict;
  fragment.querySelector("[data-field='decision_note']").textContent = data.decision_note;
  fragment.querySelector("[data-field='hybrid_percent']").textContent = formatPercent(data.scores.hybrid);
  fragment.querySelector("[data-field='risk_marker']").style.left = `${Math.min(100, Math.max(0, Number(data.scores.hybrid || 0) * 100))}%`;

  fragment.querySelector("[data-field='tfidf_score']").textContent = evidenceBand(data.scores.tfidf);
  fragment.querySelector("[data-field='sbert_score']").textContent = evidenceBand(data.scores.sbert);
  fragment.querySelector("[data-field='hybrid_score']").textContent = formatPercent(data.scores.hybrid);
  fragment.querySelector("[data-field='coverage_score']").textContent = formatPercent(data.features.coverage);
  fragment.querySelector("[data-field='hybrid_prediction']").textContent = predictionText(data.predictions.hybrid, data.thresholds.hybrid);

  makeKeyValueList(fragment.querySelector("#feature-list"), [
    ["Final risk", `${formatPercent(data.scores.hybrid)} (${evidenceBand(data.scores.hybrid)})`],
    ["Text overlap", `${formatPercent(data.features.global_tfidf)} (${evidenceBand(data.features.global_tfidf)})`],
    ["Meaning similarity", `${formatPercent(data.features.global_sbert)} (${evidenceBand(data.features.global_sbert)})`],
    ["Sentence evidence", `${formatPercent(data.scores.local_signal)} (${evidenceBand(data.scores.local_signal)})`],
    ["Strongest sentence match", `${formatPercent(data.features.peak_local)} (${evidenceBand(data.features.peak_local)})`],
    ["Average top matches", `${formatPercent(data.features.mean_top_local)} (${evidenceBand(data.features.mean_top_local)})`],
    ["Coverage", formatPercent(data.features.coverage)],
  ]);

  makeKeyValueList(fragment.querySelector("#summary-list"), [
    ["Verdict", data.verdict],
    ["Plain explanation", explainRisk(data)],
    ["Decision rule", `${formatPercent(data.thresholds.hybrid)} or higher is suspicious`],
    ["Exact ratio", formatPercent(data.features.exact_ratio)],
    ["Strong matches", data.features.match_count],
    ["Source sentences", data.features.source_sentence_count],
    ["Decision", data.predictions.hybrid ? "Suspicious" : "Less suspicious"],
  ]);

  renderMatches(fragment.querySelector("#matches-list"), data.top_matches || []);

  reportPanel.classList.remove("is-empty");
  reportPanel.classList.add("has-report");
  reportPanel.innerHTML = "";
  reportPanel.append(fragment);
}

loadSampleButton.addEventListener("click", () => {
  const selectedIndex = Number(sampleSelect.value || 0);
  const sample = samples[selectedIndex] || samples[0];
  sourceText.value = sample.source;
  suspiciousText.value = sample.suspicious;
  setStatus(`${sample.title} loaded. Run the model to generate a report.`);
});

samples.forEach((sample, index) => {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = sample.title;
  sampleSelect.append(option);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const source = sourceText.value.trim();
  const suspicious = suspiciousText.value.trim();

  if (source.length < 20 || suspicious.length < 20) {
    setStatus("Please enter at least 20 characters in both boxes.", true);
    return;
  }

  const button = form.querySelector("button[type='submit']");
  button.disabled = true;
  button.classList.add("is-loading");
  setStatus("Generating detailed plagiarism report...");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_text: source,
        suspicious_text: suspicious,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    renderReport(data);
    setStatus("Report generated.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
  }
});
