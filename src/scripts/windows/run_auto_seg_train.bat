cd ..\..
set /p commit_message=Enter commit message: 

if "%commit_message%"=="" (
    echo Commit message cannot be empty. Exiting...
    exit /b
)

python runner.py autoencoder_segmentation train 10
python runner.py autoencover_segmentation test

cd ..

git add .
git commit -m "%commit_message%"
git push origin main

shutdown -s -f -t 0