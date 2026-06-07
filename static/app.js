const els = {
  start: document.querySelector("#startBtn"),
  stop: document.querySelector("#stopBtn"),
  reset: document.querySelector("#resetBtn"),
  video: document.querySelector("#videoFeed"),
  empty: document.querySelector("#emptyState"),
  currentPose: document.querySelector("#currentPose"),
  message: document.querySelector("#message"),
  confidence: document.querySelector("#confidence"),
  holdTime: document.querySelector("#holdTime"),
  stabilityValue: document.querySelector("#stabilityValue"),
  stabilityBar: document.querySelector("#stabilityBar"),
  balanceRing: document.querySelector("#balanceRing"),
  balanceScore: document.querySelector("#balanceScore"),
  balanceStatus: document.querySelector("#balanceStatus"),
  poseCount: document.querySelector("#poseCount"),
  totalTime: document.querySelector("#totalTime"),
  avgBalance: document.querySelector("#avgBalance"),
  startedAt: document.querySelector("#startedAt"),
  rows: document.querySelector("#sessionRows"),
};

function titlePose(value) {
  if (!value) return "Ready";
  return value;
}

async function postJSON(url) {
  const response = await fetch(url, { method: "POST" });
  return response.json();
}

async function refreshState() {
  const response = await fetch("/api/session");
  const state = await response.json();
  renderState(state);
}

function renderState(state) {
  els.currentPose.textContent = titlePose(state.display_pose || state.current_pose);
  els.message.textContent = state.message || "";
  els.confidence.textContent = `${state.confidence || 0}%`;
  els.holdTime.textContent = `${state.accumulated_time || 0}s`;

  const progress = Math.round((state.stability_progress || 0) * 100);
  els.stabilityValue.textContent = `${progress}%`;
  els.stabilityBar.style.width = `${progress}%`;

  if (state.balance === null || state.balance === undefined) {
    els.balanceScore.textContent = "--";
    els.balanceRing.style.setProperty("--score", 0);
  } else {
    els.balanceScore.textContent = Math.round(state.balance);
    els.balanceRing.style.setProperty("--score", state.balance);
  }
  els.balanceStatus.textContent = state.balance_status || "Collecting";

  els.poseCount.textContent = state.session_log.length;
  els.totalTime.textContent = `${state.total_time || 0}s`;
  els.avgBalance.textContent = state.average_balance === null ? "--" : Math.round(state.average_balance);
  els.startedAt.textContent = state.started_at ? `Started ${state.started_at}` : "Not started";
  els.empty.classList.toggle("hidden", state.active);
  els.video.classList.toggle("visible", state.active);
  els.start.disabled = state.active;
  els.stop.disabled = !state.active;

  renderRows(state.session_log);
}

function renderRows(rows) {
  if (!rows.length) {
    els.rows.innerHTML = '<tr><td colspan="4" class="empty-row">No poses logged yet.</td></tr>';
    return;
  }

  els.rows.innerHTML = rows.slice().reverse().map(entry => `
    <tr>
      <td>${titlePose(entry.pose)}</td>
      <td>${entry.duration}s</td>
      <td>${entry.balance === null || entry.balance === undefined ? "--" : `${entry.balance}%`}</td>
      <td>${entry.logged_at}</td>
    </tr>
  `).join("");
}

els.start.addEventListener("click", async () => {
  els.video.src = `/video_feed?ts=${Date.now()}`;
  renderState(await postJSON("/api/start"));
});

els.stop.addEventListener("click", async () => {
  renderState(await postJSON("/api/stop"));
});

els.reset.addEventListener("click", async () => {
  renderState(await postJSON("/api/reset"));
});

refreshState();
setInterval(refreshState, 500);
