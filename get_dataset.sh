mkdir -p tgz
cd tgz

# download
for i in `cat ../data_urls.txt`; do echo $i; wget $i; done

# Unpack
for f in ./*.tar.gz; do gunzip $i.tar.gz; tar -xvf $i.tar; rm $i.tar; done

# Relocate
mkdir -p ../matrix
for i in `find . -name *.mtx`; do echo $i; mv $i ../matrix/; done

# Cleanup
cd ..
rm -rf tgz
