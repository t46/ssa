# WVS Analysis Pipeline Makefile

.PHONY: all research analysis paper pdf clean help

# Default target
all: research analysis paper pdf
	@echo "✅ Full pipeline completed!"

# Step 3: Generate research hypotheses
research:
	@echo "Running Step 3: Generate research hypotheses..."
	uv run python src/generate_research_ideas.py

# Step 4: Run statistical analysis
analysis: research
	@echo "Running Step 4: Execute statistical analyses..."
	uv run python src/analysis.py

# Step 5: Generate LaTeX paper
paper: analysis
	@echo "Running Step 5: Generate LaTeX paper..."
	uv run python src/generate_paper.py

# Compile PDF from LaTeX
pdf: paper
	@echo "Compiling LaTeX to PDF..."
	cd outputs && pdflatex -interaction=nonstopmode paper.tex
	cd outputs && pdflatex -interaction=nonstopmode paper.tex

# Run the full pipeline using Python script
run:
	uv run python run_pipeline.py

# Clean generated files
clean:
	rm -f spec/research.yaml
	rm -rf outputs/*
	@echo "✅ Cleaned generated files"

# Help
help:
	@echo "WVS Analysis Pipeline"
	@echo "===================="
	@echo "Available targets:"
	@echo "  make all      - Run full pipeline (steps 3-5 + PDF)"
	@echo "  make research - Generate research hypotheses only"
	@echo "  make analysis - Run analysis (requires research)"
	@echo "  make paper    - Generate paper (requires analysis)"
	@echo "  make pdf      - Compile PDF (requires paper)"
	@echo "  make run      - Run full pipeline using Python script"
	@echo "  make clean    - Remove all generated files"
	@echo "  make help     - Show this help message"