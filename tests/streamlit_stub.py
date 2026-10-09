
class _Recorder:
    """Collects everything the page writes, so tests can inspect it."""
    def __init__(self):
        self.calls = []
        self.text_inputs = []
        self.submit = False
        self._text_area_index = 0
        self.bar_chart_data = None
        self.progress_values = []
        self.uploaded_file = None
        self.buttons = {}
        self.downloads = []
        self.tables = []
    def reset(self):
        """Start a brand-new session: cleared widgets AND cleared state."""
        session_state.clear()
        self.calls = []
        self._text_area_index = 0
        self.bar_chart_data = None
        self.progress_values = []
        self.uploaded_file = None
        self.buttons = {}
        self.downloads = []
        self.tables = []
    def record(self, kind, *args):
        self.calls.append((kind, args))
    def rendered_text(self):
        parts = []
        for _, args in self.calls:
            for arg in args:
                if isinstance(arg, str):
                    parts.append(arg)
        return "\n".join(parts)
recorder = _Recorder()
class _SessionState(dict):
    """
    Stand-in for st.session_state: a dict that also allows attribute access.
    It deliberately survives a script rerun (a fresh import of app.py) in the
    same way the real session state does, which is what lets the tests prove
    that batch results are not lost when the page reruns.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
    def __setattr__(self, name, value):
        self[name] = value
session_state = _SessionState()
class _Context:
    """Stands in for st.spinner(...), st.container(...), st.expander(...)."""
    def __enter__(self):
        return self
    def __exit__(self, *exc_info):
        return False
class _Column(_Context):
    def metric(self, label, value, delta=None):
        recorder.record("metric", label, str(value), str(delta) if delta is not None else "")
class _Form(_Context):
    pass
def set_page_config(**kwargs):
    recorder.record("set_page_config")
def cache_resource(*decorator_args, **decorator_kwargs):
    """st.cache_resource used as @st.cache_resource(show_spinner=False)."""
    def decorator(func):
        return func
    if decorator_args and callable(decorator_args[0]):
        return decorator_args[0]
    return decorator
def title(text):
    recorder.record("title", text)
def caption(text):
    recorder.record("caption", text)
def header(text):
    recorder.record("header", text)
def subheader(text):
    recorder.record("subheader", text)
def markdown(text, **kwargs):
    recorder.record("markdown", text)
def write(*args, **kwargs):
    recorder.record("write", *[str(a) for a in args])
def success(text):
    recorder.record("success", text)
def warning(text):
    recorder.record("warning", text)
def error(text):
    recorder.record("error", text)
def info(text):
    recorder.record("info", text)
def metric(label, value, delta=None):
    recorder.record("metric", label, str(value), str(delta) if delta is not None else "")
def bar_chart(data=None, x=None, y=None, **kwargs):
    """Stands in for st.bar_chart(records, x=..., y=...). Records the
    plotted labels and values (no pandas involved) so tests can assert the
    comparison chart actually received all four dimensions."""
    labels = [str(row[x]) for row in data] if x else []
    values = [row[y] for row in data] if y else []
    recorder.record("bar_chart", *labels)
    recorder.bar_chart_data = {"labels": labels, "values": values}
def spinner(text=""):
    recorder.record("spinner", text)
    return _Context()
def container(**kwargs):
    return _Context()
def expander(label, **kwargs):
    recorder.record("expander", label)
    return _Context()
def columns(count):
    recorder.record("columns", str(count))
    number = count if isinstance(count, int) else len(count)
    return [_Column() for _ in range(number)]
def form(key):
    recorder.record("form", key)
    return _Form()
def text_area(label, placeholder="", height=None):
    recorder.record("text_area", label)
    index = recorder._text_area_index
    recorder._text_area_index += 1
    if index < len(recorder.text_inputs):
        return recorder.text_inputs[index]
    return ""
def form_submit_button(label):
    recorder.record("form_submit_button", label)
    return recorder.submit
def stop():
    raise RuntimeError("st.stop() was called")
class _Sidebar(_Context):
    """st.sidebar supports both `with st.sidebar:` and st.sidebar.write(...)."""
    header = staticmethod(header)
    write = staticmethod(write)
    caption = staticmethod(caption)
    markdown = staticmethod(markdown)
    error = staticmethod(error)
    expander = staticmethod(expander)
sidebar = _Sidebar()
class _Empty(_Context):
    """Stands in for the placeholder returned by st.empty()."""
    caption = staticmethod(caption)
    write = staticmethod(write)
    markdown = staticmethod(markdown)
class _Progress:
    """Stands in for st.progress(); records every real progress value."""
    def progress(self, value, text=None):
        recorder.progress_values.append(value)
        recorder.record("progress", str(value))
def progress(value=0.0, text=None):
    recorder.progress_values.append(value)
    recorder.record("progress", str(value))
    return _Progress()
def empty():
    return _Empty()
def file_uploader(label, type=None, **kwargs):
    recorder.record("file_uploader", label)
    return recorder.uploaded_file
def button(label, **kwargs):
    recorder.record("button", label)
    return recorder.buttons.get(label, False)
def download_button(label, data=None, file_name=None, mime=None, **kwargs):
    recorder.record("download_button", label)
    recorder.downloads.append({"label": label, "data": data, "file_name": file_name})
    return False
def dataframe(data=None, **kwargs):
    recorder.record("dataframe")
    recorder.tables.append(data)
def table(data=None, **kwargs):
    recorder.record("table")
    recorder.tables.append(data)
def divider():
    recorder.record("divider")
class _UploadedFile:
    """Minimal stand-in for a Streamlit UploadedFile."""
    def __init__(self, content, name="batch.csv"):
        self._content = content if isinstance(content, bytes) else content.encode("utf-8")
        self.name = name
    def read(self):
        return self._content
_Recorder.progress_values = []
_Recorder.uploaded_file = None
_Recorder.buttons = {}
_Recorder.downloads = []
recorder.progress_values = []
recorder.uploaded_file = None
recorder.buttons = {}
recorder.downloads = []
def radio(label, options, index=0, **kwargs):
    """Navigation stub: return None so the app renders every page in one run,
    letting the smoke tests exercise all views of the sidebar layout."""
    recorder.calls.append(("radio", label))
    return None
def text_input(label, value="", **kwargs):
    recorder.calls.append(("text_input", label))
    return value
def code(text, **kwargs):
    recorder.calls.append(("code", text))
def multiselect(label, options, default=None, **kwargs):
    """Milestone 4 filter widget: returns the default selection unchanged."""
    recorder.calls.append(("multiselect", label))
    return list(default) if default is not None else []
def selectbox(label, options, index=0, **kwargs):
    recorder.calls.append(("selectbox", label))
    options = list(options)
    if not options:
        return None
    return options[index if 0 <= index < len(options) else 0]
def slider(label, min_value=0.0, max_value=100.0, value=None, step=None, **kwargs):
    recorder.calls.append(("slider", label))
    if value is not None:
        return value
    return (min_value, max_value)
def checkbox(label, value=False, **kwargs):
    recorder.calls.append(("checkbox", label))
    return bool(recorder.buttons.get(label, value))
def line_chart(data=None, x=None, y=None, **kwargs):
    labels = [str(row[x]) for row in data] if x and data else []
    recorder.calls.append(("line_chart", *labels))
    return None
def tabs(labels):
    for label in labels:
        recorder.calls.append(("tab", label))
    return [_Context() for _ in labels]
