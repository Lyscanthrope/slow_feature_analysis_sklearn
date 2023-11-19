.PHONY: clean data create_environment

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create codna environnement
create_environment:
	conda env update -f environment.yml

remove_environment:
	conda remove -n Accuracy_analysis

