clear_video:
	find . -type f -name "*.mp4" -exec rm {} \;

clear_models:
	find . -type f -name "*.h5" -exec rm {} \;

clear_all:
	make clear_video && make clear_models
