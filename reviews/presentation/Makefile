filename=presentation

pdf: 
	pdflatex ${filename}.tex
	bibtex $(filename)
	pdflatex ${filename}.tex
	pdflatex ${filename}.tex

pdf-print: ps
	ps2pdf -dColorConversionStrategy=/LeaveColorUnchanged -dPDFSETTINGS=/printer ${filename}.ps

text: html
	html2text -width 100 -style pretty ${filename}/${filename}.html | sed -n '/./,$$p' | head -n-2 >${filename}.txt

html:
	@#latex2html -split +0 -info "" -no_navigation ${filename}
	htlatex ${filename}


dvi:
	latex ${filename}
	bibtex ${filename}||true
	latex ${filename}
	latex ${filename}

read:
	evince ${filename}.pdf &

aread:
	acroread ${filename}.pdf

clean:
	rm -f *.ps *.log *.aux *.out *.bbl *.blg *.toc
	find . -name "*eps-converted-to*" -exec rm -rf {} \;
