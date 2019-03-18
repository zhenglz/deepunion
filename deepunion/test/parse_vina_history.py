import sys, os
import linecache
from collections import OrderedDict
import operator


class ParseVinaHistory(object):

    def __init__(self, vina_hist):
        if os.path.exists(vina_hist):
            self.history_fn = vina_hist
        else:
            self.history_fn = None
            print("Warning: %s not exist. " % self.history_fn)

        self.energies = OrderedDict()
        self.model_ln = OrderedDict()

        self.sorted_energies = None
        self.history_parsed_ = False
        self.energies_rervese = False

    def read_energies(self):
        if not self.history_parsed_:
            with open(self.history_fn) as lines:
                for i, s in enumerate(lines):
                    if "REMARK step #" in s and "energy" in s:
                        model = int(s.split("#")[1].split()[0])
                        energy= float(s.split("=")[-1])

                        ln_start = i

                        self.energies[model] = energy
                        self.model_ln[model] = ln_start

                    else:
                        pass

        self.history_parsed_ = True

        self.sort_energies(decending=self.energies_rervese)

        return self

    def get_model_lines(self, model_ids):
        lines = []

        if not self.history_parsed_:
            self.read_energies()

        for id in model_ids:
            lines_id = []
            start_ln = self.model_ln[id]
            print(id, start_ln)
            keep_lines = True
            s = ""
            for ndx in range(start_ln, start_ln+999):
                if keep_lines:
                    s = linecache.getline(self.history_fn, ndx)
                    lines_id.append(s)
                    #print(s)

                if "ENDMDL" in s:
                    keep_lines = False
                    #lines_id.append(s)
                    s = ""

            lines.append(lines_id)

        return lines

    def sort_energies(self, decending=True):

        self.sorted_energies = OrderedDict(sorted(self.energies.items(),
                                                  key=operator.itemgetter(1),
                                                  reverse=decending))

        return self

    def get_first_n_models(self, top_n=10):
        topn_lines = []

        if not self.history_parsed_:
            self.read_energies()
            #self.history_parsed_ = True

        topn_models = [x[0] for x in list(self.sorted_energies.items())[: top_n]]
        topn_lines = self.get_model_lines(topn_models)

        return topn_lines


if __name__ == "__main__":

    fn = sys.argv[1]
    fout = sys.argv[2]

    topn = 10

    vinaout = ParseVinaHistory(vina_hist=fn)
    vinaout.read_energies()

    model_lines = vinaout.get_first_n_models(topn)

    for i, ls in enumerate(model_lines):
        with open(fout+"%d.pdbqt" % i, 'w') as tofile:
            for s in ls:
                tofile.write(s)

    print("history best generated.")

