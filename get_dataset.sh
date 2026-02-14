mkdir -p tmp
cd tmp

# download
for i in `cat ../data_urls.txt`; do echo $i; wget $i; done

# Unpack
for f in ./*.tar.gz; do
  [ -f "$f" ] || continue
  echo "Unpacking $f"
  tar -xzf "$f"
done

# Relocate
for i in `find . -name *.mtx`; do echo $i; mv $i ../matrix/; done

# Cleanup
cd ..
rm -rf tmp
