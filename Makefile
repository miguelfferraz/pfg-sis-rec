DATASETS_DIR := datasets
EXTRACTED_DIR := $(DATASETS_DIR)/extracted

extract-datasets:
	@echo "Criando diretório de extração..."
	@mkdir -p $(EXTRACTED_DIR)
	@echo "Extraindo arquivos .zip dos datasets..."
	@for zipfile in $(DATASETS_DIR)/*.zip; do \
		if [ -f "$$zipfile" ]; then \
			filename=$$(basename "$$zipfile" .zip); \
			echo "Extraindo $$zipfile para $(EXTRACTED_DIR)/"; \
			unzip -q "$$zipfile" -d "$(EXTRACTED_DIR)/"; \
		fi; \
	done
	@echo "Extração concluída!"

clean-extracted:
	@echo "Removendo arquivos extraídos..."
	@rm -rf $(EXTRACTED_DIR)
	@echo "Arquivos extraídos removidos!"
