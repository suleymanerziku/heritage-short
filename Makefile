PYTHON ?= python

.PHONY: all validate smoke radio film costs narration radio_cut keyframes shots music assemble help

help:
	@echo "Available targets:"
	@echo "  make validate  - Validate storyboard.json against schema/storyboard.schema.json"
	@echo "  make smoke     - Run pipeline/smoke_test.py"
	@echo "  make radio     - Run narration and radio cut assembly"
	@echo "  make film      - Run validate, narration, keyframes, shots, music, assemble in order"
	@echo "  make costs     - Run pipeline/report_costs.py to update COSTS.md"

validate:
	$(PYTHON) -c "import json, sys, os, subprocess; subprocess.run([sys.executable, 'pipeline/validate_storyboard.py'], check=True) if os.path.exists('pipeline/validate_storyboard.py') else (jsonschema.validate(instance=json.load(open('storyboard.json')), schema=json.load(open('schema/storyboard.schema.json'))) or print('Validation passed: storyboard.json matches schema/storyboard.schema.json')) if __import__('jsonschema') else sys.exit(1)"

smoke:
	$(PYTHON) pipeline/smoke_test.py

narration:
	$(PYTHON) pipeline/narration.py

radio_cut:
	$(PYTHON) pipeline/radio_cut.py

radio: narration radio_cut

keyframes:
	$(PYTHON) pipeline/keyframes.py

shots:
	$(PYTHON) pipeline/shots.py

music:
	$(PYTHON) pipeline/music.py

assemble:
	$(PYTHON) pipeline/assemble.py

film: validate narration keyframes shots music assemble

costs:
	$(PYTHON) pipeline/report_costs.py
