# IAEW-Masterarbeit -- Arbeitsstand

Dieses Verzeichnis basiert unmittelbar auf der bereitgestellten IAEW-LaTeX-Vorlage.
Seitenlayout, Titelseite, Erklärung, Kopfzeilen, Typografie, Literaturstil und RWTH-Elemente
werden weiterhin durch die Vorlagendateien bestimmt.

Kompilierung in diesem Verzeichnis:

```text
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Vor der Abgabe müssen alle Felder in eckigen Klammern in
`extras/personal_settings.tex` durch die offiziellen persönlichen Daten ersetzt werden.

Das veraltete Paket `units` ist in `extras/header.tex` deaktiviert, da es mit dem aktuellen
`siunitx` unter TeX Live 2024 kollidiert. Diese Kompatibilitätsanpassung verändert das
IAEW-Seitenlayout nicht.

Die numerischen Hauptergebnisse stammen aus `forecast_mpc/comparison_6days`. Die Dateien in
`images/results` sind Publikationskopien; verbindlich bleiben die ursprünglichen
Ergebnisverzeichnisse.
