# MLCS Improvement Suggestions

> Analysis performed on 2026-01-07

---

## 1. Code Quality & Architecture

| Area | Issue | Suggestion |
|------|-------|------------|
| **Error Handling** | Basic try/except without retry logic | Add exponential backoff retries using `tenacity` library |
| **Configuration** | Hardcoded model names and parameters | Create `config.py` or use environment variables |
| **Typing** | Incomplete type annotations | Add return types to all functions |
| **Testing** | No tests present | Add `pytest` tests for graph nodes and state management |

---

## 2. Performance Optimizations

- **Token limits**: Add `max_tokens` and `timeout` parameters to model calls
- **True streaming**: Use `astream_events` for token-by-token streaming in UI
- **Caching**: Add response caching using `@st.cache_resource`

---

## 3. Feature Additions

| Feature | Description | Priority |
|---------|-------------|----------|
| **Model Selection UI** | Let users pick which models participate | High |
| **Round Control** | Configurable number of debate rounds (currently hardcoded to 3) | High |
| **Export Results** | Add PDF/Markdown export of debate history | Medium |
| **Conversation History** | Store and replay past debates | Medium |
| **Temperature Control** | Allow users to adjust model creativity | Low |

---

## 4. UI/UX Improvements

- [ ] Show actual model names (`llama3.2`) instead of `model_a` in round display
- [ ] Add progress bar showing current round / total rounds
- [ ] Use `st.code()` for code responses (syntax highlighting)
- [ ] Add dark mode toggle

---

## 5. Robustness & Reliability

- [ ] Add health check to verify Ollama is running before starting
- [ ] Implement graceful degradation (continue if one model fails)
- [ ] Add model availability check function

---

## 6. Documentation & Maintainability

- [ ] Add docstrings to all functions in `graph.py`
- [ ] Create `ARCHITECTURE.md` explaining the LangGraph flow
- [ ] Add inline comments for complex regex patterns

---

## 7. Quick Wins (Low Effort, High Impact)

### Fix model name display in UI (`ui.py:70`):
```python
st.markdown(f"**{MODEL_NAMES.get(agent_name, agent_name)}**")
```

### Add configurable round count (sidebar):
```python
num_rounds = st.sidebar.slider("Debate Rounds", 2, 5, 3)
```

### Pin dependency versions (`requirements.txt`):
```
langgraph>=0.2.0
streamlit>=1.30.0
```
