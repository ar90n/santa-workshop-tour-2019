mkdir -p ../input/santa2019work/
poetry run kaggle d download ar90ngas/santa2019work -p ../input/santa2019work/
unzip ../input/santa2019work/santa2019work.zip -d ../input/santa2019work


mkdir -p ../input/santa-workshop-tour-2019
poetry run kaggle c download -c santa-workshop-tour-2019 -p ../input/santa-workshop-tour-2019
unzip ../input/santa-workshop-tour-2019/santa-workshop-tour-2019.zip -d ../input/santa-workshop-tour-2019
