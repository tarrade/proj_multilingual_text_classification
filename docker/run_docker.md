# How to run docker container

## Create the docker container locally
```source build_image_locally.sh```   

## Run the docker container locally
```docker run $IMAGE_URI --verbosity_level=DEBUG```

other way when using CMD instead of ENTRYPOINT:   

```docker run image_id``` 
```docker run --rm image_id python -m model.test.task --verbosity_level=DEBUG``` 