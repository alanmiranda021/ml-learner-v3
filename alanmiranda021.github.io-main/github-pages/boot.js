const boot = document.getElementById('boot');
const root = document.getElementById('root');
const tip = document.getElementById('tip');
let finished = false;
const tellParent = (type) => window.parent.postMessage({ type }, window.location.origin);
const slowTimer = setTimeout(() => {
  tip.textContent = 'O ambiente ainda está carregando. A primeira abertura depende da conexão e pode levar alguns minutos.';
}, 60000);
const observer = new MutationObserver(() => {
  if (root.querySelector('[data-testid="stFileUploader"]')) {
    finished = true;
    clearTimeout(slowTimer);
    observer.disconnect();
    boot.classList.add('done');
    setTimeout(() => boot.remove(), 800);
    tellParent('ml-learner-ready');
  }
});
observer.observe(root, { childList: true, subtree: true });
function failure(error) {
  if (finished) return;
  finished = true;
  clearTimeout(slowTimer);
  observer.disconnect();
  document.querySelector('.bar').hidden = true;
  tip.textContent = 'Não foi possível iniciar o laboratório. Confira sua conexão ou tente novamente. Você também pode executar a versão Python local.';
  const retry = document.createElement('button');
  retry.textContent = 'Tentar novamente';
  retry.addEventListener('click', () => window.location.reload());
  const guide = document.createElement('a');
  guide.href = './LEIAME.md';
  guide.textContent = 'Consultar guia local';
  boot.append(retry, guide);
  console.error('ML Learner:', error);
  tellParent('ml-learner-error');
}
window.addEventListener('unhandledrejection', (event) => failure(event.reason));
try {
  if (window.location.protocol === 'file:') throw new Error('Use um servidor HTTP para abrir a aplicação.');
  // Fetch and validate the project before starting the Python environment.
  const paths = ['app.py', 'core/env.py', 'core/io.py',
    'core/preprocess.py', 'core/models.py', 'core/evaluate.py', 'core/auto.py',
    'core/report_pdf.py', 'core/predict.py'];
  const files = Object.fromEntries(await Promise.all(paths.map(async (path) => {
    const response = await fetch(new URL(path, import.meta.url));
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return [path, await response.text()];
  })));
  // Empty package marker: no network request needed (Jekyll may hide underscored files).
  files['core/__init__.py'] = '';
  tip.textContent = 'Baixando Python e os pacotes de análise…';
  const { mount } = await import('https://cdn.jsdelivr.net/npm/@stlite/browser@0.85.1/build/stlite.js');
  await mount({
    requirements: ['scikit-learn', 'scipy', 'pandas', 'numpy', 'plotly', 'openpyxl', 'packaging', 'reportlab', 'joblib'],
    entrypoint: 'app.py', files,
    streamlitConfig: {
      'theme.base': 'dark', 'theme.primaryColor': '#25e7ff',
      'theme.backgroundColor': '#06101a', 'theme.secondaryBackgroundColor': '#102331',
      'theme.textColor': '#e1edf1', 'client.toolbarMode': 'viewer',
    },
  }, root);
} catch (error) { failure(error); }
