VENV=./.venv/bin

.PHONY: render-demo concat-demo ui

ui:
	$(VENV)/python gradio_production_ui.py

render-demo:
	$(VENV)/python tools/channel_manager.py render Demo \
		--script scripts/Demo/demo.csv \
		--voice demo \
		--to-48k --lufs-target -16 --overwrite

concat-demo:
	$(VENV)/python tools/concat.py \
		--input-dir outputs/Demo \
		--out outputs/Demo/combined.wav \
		--gap-seconds 0.5 --target-sr 48000 --mono --lufs-target -16
