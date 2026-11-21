'use strict';

const localMail = (() => {
  const draftKey = 'ayai-mail:local:draft:1';
  const timeFormat = new Intl.DateTimeFormat('ru-RU', {dateStyle: 'short', timeStyle: 'short'});
  let active = false, generation = 0, accounts = [], accountsKnown = false;
  let frequency = 0, listError = '', timer = 0, loading = null;
  let connecting = false, sending = false, fromComposer = false;
  let desiredSender = '', retry = null, review = null, refreshRunning = false, refreshAgain = false;
  const syncing = new Set();

  function today() {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }
  function feedback(id, message, error = false) {
    const el = byId(id);
    el.textContent = message;
    el.hidden = !message;
    el.classList.toggle('is-error', error);
  }
  function saveDraft() {
    if (!active) return;
    const draft = {account: desiredSender, to: byId('compose-to').value, subject: byId('compose-subject').value,
      body: byId('compose-body').value, retry, review};
    try {
      if (draft.to || draft.subject || draft.body || retry) sessionStorage.setItem(draftKey, JSON.stringify(draft));
      else sessionStorage.removeItem(draftKey);
    } catch { /* The open tab still holds the draft if storage is unavailable. */ }
  }
  function restoreDraft() {
    try {
      const draft = JSON.parse(sessionStorage.getItem(draftKey));
      if (!draft) return;
      for (const field of ['to', 'subject', 'body']) {
        byId('compose-' + field).value = typeof draft[field] === 'string' ? draft[field] : '';
      }
      desiredSender = typeof draft.account === 'string' ? draft.account : '';
      if (typeof draft.retry?.signature === 'string' && /^[a-f0-9-]{36}$/i.test(draft.retry.id)) retry = draft.retry;
      if (typeof draft.review?.content === 'string' && ['server', 'external'].includes(draft.review.source)) review = draft.review;
    } catch { /* Empty or unavailable tab storage. */ }
  }
  function uuid() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const b = crypto.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 15) | 64;
    b[8] = (b[8] & 63) | 128;
    const h = Array.from(b, n => n.toString(16).padStart(2, '0')).join('');
    return [h.slice(0, 8), h.slice(8, 12), h.slice(12, 16), h.slice(16, 20), h.slice(20)].join('-');
  }
  async function request(path, body, controller = new AbortController()) {
    if (!active || state.demo || apiOrigin !== location.origin) throw new Error('Локальная почта недоступна.');
    const opts = {mode: 'same-origin', credentials: 'same-origin', cache: 'no-store', redirect: 'error', signal: controller.signal};
    if (body !== undefined) {
      opts.method = 'POST';
      opts.headers = {'Content-Type': 'application/json', 'X-Mailsearch-Local': '1'};
      opts.body = JSON.stringify(body);
    }
    const timeout = setTimeout(() => controller.abort(), body === undefined ? 15000 : 90000);
    try {
      let response;
      try { response = await fetch('/api/local/' + path, opts); }
      catch { throw new Error('Локальный сервер не ответил. Проверьте подключение и повторите попытку.'); }
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = typeof data?.detail === 'string' ? data.detail :
          Array.isArray(data?.detail) ? data.detail.map(x => x.msg).filter(Boolean).join('. ') : '';
        const error = new Error(detail || 'Локальный сервер вернул ошибку ' + response.status + '.');
        error.status = response.status;
        throw error;
      }
      if (!data) throw new Error('Локальный сервер прислал неполный ответ.');
      return data;
    } finally { clearTimeout(timeout); }
  }
  function hasAccount(id) { return accounts.some(a => a.id === id); }
  function letterFields() {
    const to = [...new Set(byId('compose-to').value.split(',').map(s => s.trim()).filter(Boolean))];
    return {to, subject: byId('compose-subject').value, body: byId('compose-body').value};
  }
  function checkRequired(fields) {
    return Boolean(review && review.content === JSON.stringify(fields));
  }
  function requireReview(source, fields) {
    review = {source, content: JSON.stringify(fields)};
    byId('compose-checked').checked = false;
    saveDraft();
    setSendControls();
  }
  function externalLink({to, subject, body}) {
    const encode = value => encodeURIComponent(value.toWellFormed ? value.toWellFormed() : value);
    const addresses = to.map(address => encode(address).replace('%40', '@')).join(',');
    const params = [];
    if (subject) params.push('subject=' + encode(subject));
    if (body) params.push('body=' + encode(body.replace(/\r\n|\r|\n/g, '\r\n')));
    return 'mailto:' + addresses + (params.length ? '?' + params.join('&') : '');
  }
  function setSendControls() {
    const sender = byId('compose-from');
    const fields = letterFields(), showReview = !sending && checkRequired(fields);
    const checked = byId('compose-checked');
    checked.disabled = sending;
    if (!showReview) checked.checked = false;
    byId('compose-review').hidden = !showReview;
    if (showReview) byId('compose-review-hint').textContent = review.source === 'server' ?
      'Результат отправки неизвестен. Перед повтором проверьте папку «Отправленные» в своей почте.' :
      'Вы открывали это письмо в почтовой программе. Перед повтором проверьте папку «Отправленные».';
    const guarded = showReview && !checked.checked;
    sender.disabled = sending || !accounts.length;
    for (const field of ['to', 'subject', 'body']) byId('compose-' + field).readOnly = sending;
    byId('compose-send').disabled = sending || guarded || !hasAccount(sender.value);
    byId('compose-send').querySelector('span').textContent = sending ? 'Отправляем…' : 'Отправить';
    const external = byId('compose-external');
    if (active && !sending && !guarded && byId('compose-to').validity.valid && fields.to.length) {
      external.href = externalLink(fields);
      external.removeAttribute('aria-disabled');
      external.removeAttribute('tabindex');
    } else {
      external.removeAttribute('href');
      external.setAttribute('aria-disabled', 'true');
      external.tabIndex = -1;
    }
    byId('compose-connect').hidden = !accountsKnown && !listError;
    byId('compose-connect').disabled = sending;
    byId('compose-manage').hidden = !accounts.length;
    byId('compose-manage').disabled = sending;
    byId('compose-account-note').hidden = accounts.length > 0;
    byId('compose-account-hint').textContent = accountsKnown ? 'Для отправки из этого окна подключите свой ящик.' :
      listError ? 'Не удалось загрузить адреса отправителей.' : 'Загружаем адреса отправителей…';
  }
  function renderSenders() {
    const select = byId('compose-from');
    if (!desiredSender) desiredSender = hasAccount(state.mailbox) ? state.mailbox : (accounts[0]?.id || '');
    const options = document.createDocumentFragment();
    if (!hasAccount(desiredSender)) {
      const placeholder = node('option', '', accounts.length ? 'Выберите адрес отправителя' : 'Нет подключённых адресов');
      placeholder.value = '';
      options.append(placeholder);
    }
    for (const a of accounts) {
      const option = node('option', '', a.address);
      option.value = a.id;
      options.append(option);
    }
    select.replaceChildren(options);
    select.value = hasAccount(desiredSender) ? desiredSender : '';
    setSendControls();
  }
  function openCompose() {
    if (!active) return;
    if (!byId('compose-to').value && !byId('compose-subject').value && !byId('compose-body').value && !retry && hasAccount(state.mailbox)) {
      desiredSender = state.mailbox;
    }
    renderSenders();
    if (!byId('compose-dialog').open) byId('compose-dialog').showModal();
    byId('compose-to').focus();
    if (!accountsKnown) loadAccounts();
  }
  async function send(event) {
    event.preventDefault();
    if (!active || sending || !byId('compose-form').reportValidity()) return;
    const account = byId('compose-from').value;
    if (!hasAccount(account)) return;
    const fields = letterFields(), {to, subject, body} = fields;
    if (checkRequired(fields) && !byId('compose-checked').checked) return;
    if (!subject.trim() && !body.trim()) {
      feedback('compose-feedback', 'Добавьте тему или текст письма.', true);
      byId('compose-subject').focus();
      return;
    }
    const payload = {account_id: account, ...fields};
    const signature = JSON.stringify(payload);
    if (retry?.signature !== signature) retry = {signature, id: uuid()};
    // An interrupted tab may never receive the delivery receipt.
    review = {source: 'server', content: JSON.stringify(fields)};
    saveDraft();
    const version = generation;
    sending = true;
    setSendControls();
    feedback('compose-feedback', 'Отправляем письмо…');
    try {
      const data = await request('send', {...payload, request_id: retry.id});
      if (version !== generation) return;
      if (data.status !== 'sent') throw new Error('Сервер не подтвердил отправку. Черновик сохранён.');
      retry = null;
      review = null;
      for (const field of ['to', 'subject', 'body']) byId('compose-' + field).value = '';
      saveDraft();
      feedback('compose-feedback', '');
      byId('compose-dialog').close();
      personal.toast('Письмо отправлено', to.join(', '));
    } catch (error) {
      if (version !== generation) return;
      if (!error.status || error.status === 409 || (error.status >= 500 && error.status !== 502)) requireReview('server', fields);
      else { review = null; saveDraft(); }
      feedback('compose-feedback', error.message, true);
      if (!byId('compose-dialog').open) personal.toast('Отправка не подтверждена', error.message, openCompose);
    } finally {
      if (version === generation) { sending = false; setSendControls(); }
    }
  }
  function lastSync(value) {
    if (!value) return '';
    const stamp = typeof value === 'number' && value < 1e12 ? value * 1000 : value;
    const date = new Date(stamp);
    return Number.isNaN(date.getTime()) ? '' : timeFormat.format(date);
  }
  function accountHeadline(a) {
    if (a.status === 'syncing' || syncing.has(a.id)) return 'Синхронизация…';
    if (a.status === 'error') return 'Ошибка синхронизации';
    return a.last_synced ? 'Синхронизировано' : 'Ожидает синхронизации';
  }
  function renderAccounts() {
    const list = byId('accounts-list');
    const focusedId = document.activeElement?.dataset.syncAccount;
    const fragment = document.createDocumentFragment();
    for (const a of accounts) {
      const card = node('article', 'account-card');
      const badge = node('span', 'account-provider', a.provider === 'gmail' ? 'Gmail' : a.provider === 'yandex' ? 'Яндекс.Почта' : a.provider);
      const title = node('div', 'account-title');
      title.append(node('strong', '', a.address), badge);
      const phase = node('span', 'account-phase', accountHeadline(a));
      phase.classList.toggle('is-error', a.status === 'error');
      const updated = lastSync(a.last_synced);
      const meta = [mailCount(Number(a.count) || 0)];
      if (updated) meta.push('Последняя синхронизация ' + updated);
      const sync = action('Синхронизировать', 'secondary account-sync', () => syncAccount(a.id));
      sync.dataset.syncAccount = a.id;
      sync.disabled = syncing.has(a.id) || a.status === 'syncing';
      sync.setAttribute('aria-label', 'Синхронизировать ' + a.address);
      card.append(title, phase, node('p', 'account-caption', meta.join(' · ')));
      if (a.pending) card.append(node('p', 'account-pending', typeof a.pending === 'number' ?
        'В очереди на загрузку ' + mailCount(a.pending) + '.' : 'Ещё есть письма для загрузки.'));
      if (a.error) card.append(node('p', 'mail-feedback is-error', a.error));
      card.append(sync);
      fragment.append(card);
    }
    if (!accounts.length) fragment.append(node('p', 'account-caption',
      accountsKnown ? 'Пока нет подключённых аккаунтов.' : listError ? '' : 'Загружаем аккаунты…'));
    list.replaceChildren(fragment);
    if (focusedId) Array.from(list.querySelectorAll('[data-sync-account]')).find(b => b.dataset.syncAccount === focusedId && !b.disabled)?.focus();
    const frequencyText = byId('accounts-frequency');
    frequencyText.hidden = !accountsKnown;
    frequencyText.textContent = frequency > 0 ? 'Автосинхронизация каждые ' +
      (frequency % 60 === 0 ? numberFormat.format(frequency / 60) + ' мин.' : numberFormat.format(frequency) + ' сек.') :
      'Синхронизация вручную.';
    byId('accounts-load-error').hidden = !listError;
    byId('accounts-error-text').textContent = listError;
    for (const id of ['account-provider', 'account-address', 'account-password']) byId(id).disabled = connecting;
    byId('account-connect').disabled = connecting;
    byId('account-connect').textContent = connecting ? 'Подключаем…' : 'Подключить';
  }
  function schedule() {
    clearTimeout(timer);
    if (!active || document.hidden || loading || connecting || syncing.size) return;
    const busy = accounts.some(a => a.status === 'syncing' || a.pending);
    timer = setTimeout(loadAccounts, busy ? 3500 : Math.min(30, frequency > 0 ? frequency : 30) * 1000);
  }
  async function refreshBoxes() {
    if (refreshRunning) { refreshAgain = true; return; }
    const version = generation;
    refreshRunning = true;
    try {
      do {
        refreshAgain = false;
        await loadMailboxes({preserve: true});
      } while (refreshAgain && version === generation && active);
    } finally { if (version === generation) refreshRunning = false; }
  }
  function applyAccounts(list, force = false) {
    const changed = JSON.stringify(accounts) !== JSON.stringify(list);
    const wasKnown = accountsKnown;
    const hadError = Boolean(listError);
    accounts = list;
    accountsKnown = true;
    listError = '';
    if (changed || !wasKnown || hadError || force) {
      renderAccounts();
      renderSenders();
    }
    if (force || accounts.some(a => !getBox(a.id) || (Number.isFinite(a.count) && getBox(a.id).count !== a.count))) refreshBoxes();
  }
  async function loadAccounts() {
    if (!active || loading || connecting || syncing.size) return;
    clearTimeout(timer);
    const controller = new AbortController();
    loading = controller;
    const version = generation;
    try {
      const data = await request('accounts', undefined, controller);
      if (version !== generation || controller.signal.aborted) return;
      if (!Array.isArray(data.accounts)) throw new Error('Не удалось прочитать список аккаунтов.');
      const oldFrequency = frequency;
      frequency = Number(data.auto_sync_seconds) || 0;
      applyAccounts(data.accounts);
      if (oldFrequency !== frequency) renderAccounts();
      if (!accounts.length && byId('accounts-dialog').open && !byId('account-add').open && !wasAddClosed) byId('account-add').open = true;
    } catch (error) {
      if (version !== generation || loading !== controller) return;
      listError = error.message;
      renderAccounts();
      setSendControls();
    } finally {
      if (loading === controller) { loading = null; schedule(); }
    }
  }
  let wasAddClosed = false;
  function pauseList() {
    clearTimeout(timer);
    const old = loading;
    loading = null;
    old?.abort();
  }
  function upsert(account) {
    if (!account?.id || !account.address) throw new Error('Сервер не вернул подключённый аккаунт.');
    const next = [...accounts];
    const index = next.findIndex(a => a.id === account.id);
    if (index < 0) next.push(account);
    else next[index] = account;
    applyAccounts(next, true);
  }
  function openAccounts(compose = false) {
    if (!active) return;
    fromComposer = compose && byId('compose-dialog').open;
    const dialog = byId('accounts-dialog');
    if (!dialog.open) {
      wasAddClosed = false;
      byId('account-add').open = !accounts.length || fromComposer;
      feedback('accounts-feedback', '');
      dialog.showModal();
    }
    renderAccounts();
    loadAccounts();
  }
  function updateProviderHelp() {
    const yandex = byId('account-provider').value === 'yandex';
    const link = byId('account-password-link');
    link.href = yandex ? 'https://passport.yandex.ru/security/app-passwords' : 'https://myaccount.google.com/apppasswords';
    link.textContent = yandex ? 'Пароли приложений Яндекс ID' : 'Пароли приложений Google';
  }
  async function connect(event) {
    event.preventDefault();
    if (!active || connecting || !byId('account-form').reportValidity()) return;
    const provider = byId('account-provider').value, address = byId('account-address').value.trim();
    const password = byId('account-password').value;
    byId('account-password').value = '';
    const version = generation;
    connecting = true;
    pauseList();
    renderAccounts();
    feedback('accounts-feedback', 'Проверяем подключение…');
    try {
      const {account} = await request('accounts', {provider, address, password});
      if (version !== generation) return;
      upsert(account);
      byId('account-address').value = '';
      byId('account-add').open = false;
      feedback('accounts-feedback', 'Аккаунт подключён.');
      if (fromComposer && byId('compose-dialog').open) {
        desiredSender = account.id;
        renderSenders();
        saveDraft();
        byId('accounts-dialog').close();
        byId('compose-to').focus();
      }
    } catch (error) {
      if (version === generation) feedback('accounts-feedback', error.message, true);
    } finally {
      if (version === generation) { connecting = false; renderAccounts(); loadAccounts(); }
    }
  }
  async function syncAccount(id) {
    if (!active || syncing.has(id) || !hasAccount(id)) return;
    syncing.add(id);
    const version = generation;
    pauseList();
    renderAccounts();
    feedback('accounts-feedback', '');
    try {
      const {account} = await request('accounts/' + encodeURIComponent(id) + '/sync', {});
      if (version === generation) upsert(account);
    } catch (error) {
      if (version === generation) feedback('accounts-feedback', error.message, true);
    } finally {
      if (version === generation) { syncing.delete(id); renderAccounts(); loadAccounts(); }
    }
  }
  function setEnabled(value) {
    if (active === value) return;
    generation++;
    active = value;
    byId('local-controls').hidden = !active;
    if (active) {
      restoreDraft();
      renderSenders();
      loadAccounts();
      if (location.hash === '#accounts') openAccounts();
    } else {
      pauseList();
      byId('accounts-dialog').close();
      byId('compose-dialog').close();
      byId('account-password').value = '';
      byId('account-address').value = '';
      for (const f of ['to','subject','body']) byId('compose-' + f).value = '';
      accounts = []; accountsKnown = false; listError = ''; frequency = 0;
      desiredSender = ''; retry = null; review = null; connecting = false; sending = false;
      refreshRunning = false; refreshAgain = false; syncing.clear();
      byId('accounts-list').replaceChildren();
      byId('compose-from').replaceChildren();
      setSendControls();
      feedback('compose-feedback', '');
      feedback('accounts-feedback', '');
    }
  }
  function init() {
    byId('compose-open').addEventListener('click', openCompose);
    byId('accounts-open').addEventListener('click', () => openAccounts());
    byId('compose-close').addEventListener('click', () => byId('compose-dialog').close());
    byId('accounts-close').addEventListener('click', () => byId('accounts-dialog').close());
    byId('compose-connect').addEventListener('click', () => openAccounts(true));
    byId('compose-manage').addEventListener('click', () => openAccounts(true));
    byId('compose-form').addEventListener('submit', send);
    byId('compose-external').addEventListener('click', event => {
      event.preventDefault();
      const href = event.currentTarget.getAttribute('href');
      if (!active || !href) return;
      requireReview('external', letterFields());
      feedback('compose-feedback', '');
      location.assign(href);
    });
    byId('compose-form').addEventListener('input', () => {
      if (sending) return;
      feedback('compose-feedback', '');
      saveDraft();
      setSendControls();
    });
    byId('compose-from').addEventListener('change', () => {
      desiredSender = byId('compose-from').value;
      saveDraft();
      setSendControls();
    });
    byId('account-form').addEventListener('submit', connect);
    byId('account-provider').addEventListener('change', updateProviderHelp);
    updateProviderHelp();
    byId('accounts-retry').addEventListener('click', loadAccounts);
    byId('account-add').addEventListener('toggle', () => {
      if (!byId('account-add').open) { wasAddClosed = true; byId('account-password').value = ''; }
    });
    byId('accounts-dialog').addEventListener('close', () => {
      if (byId('accounts-dialog').open) return;
      fromComposer = false;
      byId('account-password').value = '';
      if (location.hash === '#accounts') history.replaceState(history.state, '', location.pathname + location.search);
    });
    window.addEventListener('hashchange', () => {
      if (location.hash === '#accounts') openAccounts();
      else if (byId('accounts-dialog').open) byId('accounts-dialog').close();
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) clearTimeout(timer);
      else if (active) loadAccounts();
    });
  }
  return {init, setEnabled, enabled: () => active, hasAccount, today};
})();
