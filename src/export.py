ExportFormat = Literal["text", "md", "json"]

def export(model, *, fmt="text", output=None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt ’{fmt}’. Expected ’text’ | ’md’ | ’json’.")
    
    if output is None:
        return text
    
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output