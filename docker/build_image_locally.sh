# create test folder
cd ../
mkdir test-docker
pwd

# copy config files localy
cp docker/test/Dockerfile test-docker/.
cp docker/entrypoint.sh test-docker/.
cp -r src test-docker/.
cp env/environment.yaml test-docker/.

cd test-docker

echo "build env variable"
# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
export IMAGE_REPO_NAME=test

# IMAGE_TAG: an easily identifiable tag for your docker image
export IMAGE_TAG=v0.0.0

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
echo $IMAGE_URI

echo "build the docker image"
#docker build -f Dockerfile .
docker build -f Dockerfile -t $IMAGE_URI ./

cd ../docker