# NBA player rookie longevity

## Create and activate python environnment

``` conda env create  -f environment.yml```  
``` conda activate nba_rookie```


## Training
Run train/training.ipynb jupyter file.
Final model is saved under deploy/

## API Deployment
 => Under deploy/ folder
### Local

 Run MLserver: 
 
```mlserver start . ```

### Under Docker container

Create Docker image : 

```mlserver build deploy/ -t deploy_nba_rookie:0.1.0 ```

Create and launch container deploy_nba_rookie:0.1.0 

``` docker run -it --rm -p 8080:8080 deploy_nba_rookie:0.1.0 ```

### Kubernetes with Kserve
**TODO**

## Inference
After API Deployment is launch:  
In infer folder, fill player_characteristic.json with values of your player  
Then run:  
``` python inference.py --input player_characteristic.json ``` 


