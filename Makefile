VENV=./.venv/bin

.PHONY: render-wiredtowork concat-wiredtowork render-wiredworkshop concat-wiredworkshop render-tinytalestv concat-tinytalestv

render-wiredtowork:
	$(VENV)/python tools/channel.py render WiredToWork \
		--script scripts/WiredToWork/first.txt \
		--voice wired-businessgirl \
		--to-48k --lufs-target -16 --overwrite

concat-wiredtowork:
	$(VENV)/python tools/concat.py \
		--input-dir outputs/WiredToWork \
		--out outputs/WiredToWork/combined.wav \
		--gap-seconds 0.5 --target-sr 48000 --mono --lufs-target -16

# WiredWorkshop (defaults to channel CSV and preset)
render-wiredworkshop:
	$(VENV)/python tools/channel.py render WiredWorkshop \
		--script $(or $(SCRIPT),scripts/WiredWorkshop/wiredworkshop.csv) \
		--voice wired-eliv3 \
		--to-48k --lufs-target -16 $(EXTRA)

concat-wiredworkshop:
	$(VENV)/python tools/concat.py \
		--input-dir outputs/WiredWorkshop \
		--out outputs/WiredWorkshop/combined.wav \
		--gap-seconds 0.5 --target-sr 48000 --mono --lufs-target -16

# TinyTalesTV (default to one CSV; override with SCRIPT=...)
render-tinytalestv:
	$(VENV)/python tools/channel.py render TinyTalesTV \
		--script $(or $(SCRIPT),scripts/TinyTalesTV/BirthdayParty.csv) \
		--to-48k --lufs-target -16 $(EXTRA)

concat-tinytalestv:
	$(VENV)/python tools/concat.py \
		--input-dir outputs/TinyTalesTV \
		--out outputs/TinyTalesTV/combined.wav \
		--gap-seconds 0.5 --target-sr 48000 --mono --lufs-target -16


