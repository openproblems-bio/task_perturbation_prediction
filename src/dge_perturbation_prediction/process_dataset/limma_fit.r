suppressMessages(library(edgeR))
suppressMessages(library(limma))

library("optparse")
option_list = list(
    make_option(
        c("--input_h5ad"),
        type="character",
    ),
    make_option(
        c("--design"),
        type="character",
    ),
    make_option(
        c("--fit_output_path"),
        type="character",
    ),
    make_option(
        c("--plot_output_path"),
        type="character",
    )
);


opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

print(opt$input_h5ad)
print(opt$design)
print(opt$fit_output_path)
print(opt$plot_output_path)

cat("> Loading H5AD\\n")
ad <- anndata::read_h5ad(opt$input_h5ad)

row.names(ad$obs) <- row.names(ad$X)

counts = data.frame(t(ad$X))
row.names(counts) <- row.names(ad$var)
colnames(counts) <- row.names(ad$obs)

cat("> Normalizing\\n")
d0 <- DGEList(counts)
d0 <- calcNormFactors(d0)


make_mm <- function(formula_str, df){
    # sets up the model matrix from the formula string and dataframe

        for (column in colnames(df)){
            assign(column, df[[column]])
            }
    mm <- model.matrix(eval(parse(text=formula_str)))
    return(mm)
    }


mm <- make_mm(opt$design, ad$obs)

if (!(length(opt$plot_output_path)==0)){
    cat("> Generating voom plot\\n")
    pdf(file = opt$plot_output_path,   # The directory you want to save the file in
        width = 4, # The width of the plot in inches
        height = 4) # The height of the plot in inches

    y <- voom(d0, mm, plot=TRUE)
    dev.off()
} else{
    y <- voom(d0, mm, plot=FALSE)
    }


cat("> Fitting...\\n")
fit <- lmFit(y, mm)

print(colnames(fit$coefficients))
if(!(length(opt$fit_output_path)==0)){
    saveRDS(fit, opt$fit_output_path)
}