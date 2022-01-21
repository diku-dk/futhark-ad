if [ ! -d venv ]
then
  python3 -m venv venv
fi
. venv/bin/activate
pip install -r requirements.txt

if [ ! -f data/random.in ]
then
  futhark dataget kmeans.fut '0' > data/random.in
fi
