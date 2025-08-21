from utils.correlazione import run_correlation_analysis
from utils.analisi_supervisionata import analisi_supervisionata
from utils.unsupervised import unsupervized



def main():
    run_correlation_analysis()
    analisi_supervisionata()
    unsupervized()


if __name__ == "__main__":
    main()

