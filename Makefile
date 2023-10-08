define PRINT_HELP_SCRIPT
while read line; do
    if [[ $$line =~ ^([a-zA-Z_-]+):.*?##\ (.*)$$ ]]; then
        target=$${BASH_REMATCH[1]}
        help=$${BASH_REMATCH[2]}
        printf "%-20s %s\n" "$$target" "$$help"
    fi
done < $(MAKEFILE_LIST)
endef
export PRINT_HELP_SCRIPT

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@/bin/bash -c "$$PRINT_HELP_SCRIPT" -- $(MAKEFILE_LIST)

virtualenv: ## Build virtualenv
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	#source venv/bin/activate

server: ## Run server
	#streamlit run app.py 
	streamlit run app_new.py 
