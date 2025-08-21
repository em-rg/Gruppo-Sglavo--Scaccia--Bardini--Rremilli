from utils.correlazione import run_correlation_analysis
from utils.analisi_supervisionata import analisi_supervisionata
from utils.unsupervised import unsupervised



def main():
    run_correlation_analysis()
    analisi_supervisionata()
    unsupervised()


if __name__ == "__main__":
    main()

