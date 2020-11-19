src=$1
timestamp=$(date +%d-%m-%Y_%H-%M-%S)
tmp_dst="img_$timestamp";
mkdir $tmp_dst;
ffmpeg -i $src $tmp_dst/img_%d.jpg -hide_banner