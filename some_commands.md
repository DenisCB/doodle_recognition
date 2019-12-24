

docker build -t my_notebook .
docker run -p 8888:8888 -v /Users/Denis/ML/hellodocker:/home/jovyan my_notebook # will use Dockerfile


docker-compose up --build

docker ps



docker exec -it doodles bash
psql -h localhost -U denis -d doodles_db -p 5432


psql -p 5432 -d postgres

docker exec -it doodles_db psql -p5432 -U denis -d doodles


# AWS deploy

ssh -i "aws_key.pem" ec2-user@ec2-35-158-1-65.eu-central-1.compute.amazonaws.com


https://hackernoon.com/running-docker-on-aws-ec2-83a14b780c56
sudo yum update
sudo yum install -y docker git
sudo service docker start
sudo usermod -aG docker ec2-user
sudo reboot

sudo curl -L https://github.com/docker/compose/releases/download/1.18.0/docker-compose-`uname -s-uname -m -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
git clone --single-branch --branch move-to-docker https://github.com/DenisCB/doodle_recognition.git
