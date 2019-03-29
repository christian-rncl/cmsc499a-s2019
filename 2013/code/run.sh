# $1 = infile (with train, test, pathway file names), $2 = reg. parameter lambda, $3 = reg. parameter sigma, $4 = class-ratio, $5 = initialization file
java -Xms2048M -Xmx6g -cp mallet.jar:mallet-deps.jar:bin pathopt.MultitaskPathOpt $1 $2 $3 $4 $5
