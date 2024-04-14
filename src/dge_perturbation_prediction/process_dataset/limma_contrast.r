suppressMessages(library(edgeR))
suppressMessages(library(limma))
suppressMessages(library(anndata))

library("optparse")
option_list = list(
    make_option(
        c("--input_fit"),
        type="character",
    ),
    make_option(
        c("--contrast"),
        type="character",
    ),
    make_option(
        c("--contrast_output_path"),
        type="character",
    )
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

cat(opt$input_fit)
cat()
cat(opt$contrast)
cat()
cat(opt$contrast_output_path)
cat()

fit = readRDS(opt$input_fit)

cat("> Computing contrast...\\n")

contr <- makeContrasts(contrasts=opt$contrast, levels=colnames(coef(fit)))
tmp <- contrasts.fit(fit, contr)
tmp <- eBayes(tmp)

result <- topTable(tmp,  n=Inf, sort='none')

if (!(length(opt$contrast_output_path)==0)){
    write.csv(result, opt$contrast_output_path)
}