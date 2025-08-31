using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

class Program
{
    static void Main()
    {
        // --- Load & parse training pairs ---
        string raw = File.ReadAllText(".\\trainData.txt");
        var pairs = TrainingData.ParsePairs(raw); // List<(user, bot)>

        // --- Build vocab (includes special tokens) ---
        var compressor = new TextCompressor(addSpecials: true);
        compressor.BuildVocabFromPairs(pairs);

        Console.WriteLine($"Vocab size: {compressor.VocabSize}");

        // --- Model ---
        var model = new TinyRNN(compressor.VocabSize, hiddenSize: 32);

        // --- Training hyperparams ---
        int epochs = 500;                // keep modest to avoid long runs
        double lr = 0.01;
        double totalLoss;

        var rng = new Random(0);

        // --- Train: one user message -> one bot message ---
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            totalLoss = 0.0;

            // Shuffle pairs each epoch (prevents fixed sequence overfitting)
            pairs.ShuffleInPlace(rng);

            foreach (var (u, b) in pairs)
            {
                // Reset state for each (User -> Bot) example
                model.ResetState();

                // Prime the RNN with the user's tokens (no loss), ending with <EOL>
                var uTokens = compressor.TokenizeToIds(u);
                foreach (int t in uTokens) model.PrimeStep(t);
                model.PrimeStep(compressor.EOL); // signal end of user message

                // Teacher-forced training on the bot's answer (+ <EOL>)
                var bTokens = compressor.TokenizeToIds(b);
                int prev = compressor.EOL; // we feed <EOL> right before bot starts
                foreach (int target in bTokens.Append(compressor.EOL))
                {
                    double newLoss = model.TrainStep(prev, target, lr);
                    totalLoss += newLoss;
                    prev = target;
                }

                // (Optional) small state noise or dropout could be added here for more variety.
            }

            Console.WriteLine($"Epoch {epoch}, Loss {totalLoss:F4}");
        }

        // --- Quick demo generation using the first training input ---
        Console.WriteLine("\n--- Demo (seeded with first training user msg) ---");
        string demoUser = pairs[0].user;
        string demoBot = GenerateReply(model, compressor, demoUser, temperature: 0.9, topK: 8, maxTokens: 40);
        Console.WriteLine($"User: {demoUser}");
        Console.WriteLine($"Bot:  {demoBot}");

        // --- Interactive chat ---
        Console.WriteLine("\n--- Chat ---");
        Console.WriteLine("type your message (or 'quit' to exit):");
        while (true)
        {
            Console.Write("User: ");
            string? user = Console.ReadLine();
            if (user == null) break;
            if (user.Trim().ToLowerInvariant() == "quit") break;

            // lowercasing to match training style
            user = user.ToLowerInvariant();

            // simple OOD/uncertainty detection: if too many tokens map to <UNK>, say we don't know
            var tokens = compressor.Tokenize(user).ToList();
            double unkRate = tokens.Count == 0 ? 1.0 :
                             tokens.Count(t => !compressor.HasWord(t)) / (double)tokens.Count;

            if (unkRate >= 0.6)
            {
                Console.WriteLine("Bot: i don't know about that yet.");
                continue;
            }

            string reply = GenerateReply(model, compressor, user, temperature: 0.9, topK: 8, maxTokens: 60);
            // guardrail: empty or mostly unknown?
            if (string.IsNullOrWhiteSpace(reply) || reply.Count(char.IsLetterOrDigit) < 2)
                reply = "i don't know about that yet.";

            Console.WriteLine("Bot: " + reply);
        }
    }

    static string GenerateReply(TinyRNN model, TextCompressor comp, string user,
                                double temperature = 0.9, int topK = 8, int maxTokens = 50)
    {
        // Reset & prime with user + <EOL>
        model.ResetState();
        foreach (int t in comp.TokenizeToIds(user)) model.PrimeStep(t);
        model.PrimeStep(comp.EOL);

        // Generate until <EOL> or maxTokens
        var outTokens = new List<int>();
        int x = comp.EOL; // first input token for bot
        for (int i = 0; i < maxTokens; i++)
        {
            int tok = model.Sample(x, temperature: temperature, topK: topK);
            if (tok == comp.EOL) break;
            outTokens.Add(tok);
            x = tok;
        }

        return comp.Detokenize(outTokens);
    }
}

// -------------------- Training data parser --------------------

static class TrainingData
{
    private static readonly Regex userRx = new(@"^\s*User:\s*(.*)$", RegexOptions.IgnoreCase);
    private static readonly Regex botRx = new(@"^\s*Bot:\s*(.*)$", RegexOptions.IgnoreCase);

    public static List<(string user, string bot)> ParsePairs(string raw)
    {
        var lines = raw.Replace("\r\n", "\n").Split('\n');
        var pairs = new List<(string user, string bot)>();

        string? curUser = null;

        foreach (var line in lines)
        {
            var u = userRx.Match(line);
            if (u.Success)
            {
                curUser = u.Groups[1].Value.Trim();
                continue;
            }

            var b = botRx.Match(line);
            if (b.Success && curUser != null)
            {
                var bot = b.Groups[1].Value.Trim();
                if (curUser.Length > 0 && bot.Length > 0)
                    pairs.Add((curUser, bot));
                curUser = null;
            }
        }

        return pairs;
    }
}

// -------------------- Tokenizer / Vocab --------------------

class TextCompressor
{
    // Specials
    public const string EOL_TOKEN = "<EOL>";
    public const string UNK_TOKEN = "<UNK>";

    private readonly Dictionary<string, int> wordToId = new();
    private readonly Dictionary<int, string> idToWord = new();
    private readonly bool addSpecials;

    public int EOL => wordToId[EOL_TOKEN];
    public int UNK => wordToId[UNK_TOKEN];

    public TextCompressor(bool addSpecials = true)
    {
        this.addSpecials = addSpecials;
        if (addSpecials)
        {
            wordToId[EOL_TOKEN] = 0; idToWord[0] = EOL_TOKEN;
            wordToId[UNK_TOKEN] = 1; idToWord[1] = UNK_TOKEN;
        }
    }

    public void BuildVocabFromPairs(IEnumerable<(string user, string bot)> pairs)
    {
        foreach (var (u, b) in pairs)
        {
            foreach (var w in Tokenize(u)) AddWord(w);
            foreach (var w in Tokenize(b)) AddWord(w);
        }
        // (EOL/UNK already present if addSpecials)
    }

    public IEnumerable<string> Tokenize(string text)
    {
        // lower-case, split on non-alphanumeric (keep apostrophes turned into spaces)
        text = text.ToLowerInvariant();
        text = Regex.Replace(text, @"[^a-z0-9]+", " ").Trim();
        if (text.Length == 0) yield break;
        foreach (var w in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            yield return w;
    }

    public int[] TokenizeToIds(string text)
        => Tokenize(text).Select(w => wordToId.TryGetValue(w, out var id) ? id : UNK).ToArray();

    public string Detokenize(IEnumerable<int> tokens)
    {
        var words = tokens.Select(t => idToWord.TryGetValue(t, out var w) ? w : UNK_TOKEN)
                          .Where(w => w != EOL_TOKEN);
        return string.Join(" ", words);
    }

    public bool HasWord(string w) => wordToId.ContainsKey(w);

    private void AddWord(string w)
    {
        if (!wordToId.ContainsKey(w))
        {
            int id = wordToId.Count;
            wordToId[w] = id;
            idToWord[id] = w;
        }
    }

    public int VocabSize => wordToId.Count;
}

// -------------------- Tiny RNN (toy) --------------------

class TinyRNN
{
    private readonly int vocabSize, hiddenSize;
    private double[,] Wx, Wh, Wy;
    private double[] h;
    private readonly Random rand = new(1);

    public TinyRNN(int vocabSize, int hiddenSize)
    {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;

        Wx = RandomMatrix(hiddenSize, vocabSize, seed: 42);
        Wh = RandomMatrix(hiddenSize, hiddenSize, seed: 43);
        Wy = RandomMatrix(vocabSize, hiddenSize, seed: 44);

        h = new double[hiddenSize];
    }

    public void ResetState() => Array.Fill(h, 0.0);

    public void PrimeStep(int x)
    {
        // forward only, no weight updates
        double[] xVec = OneHot(x, vocabSize);
        h = Tanh(Add(MatVec(Wx, xVec), MatVec(Wh, h)));
    }

    public double TrainStep(int x, int target, double lr)
    {
        // Forward
        double[] xVec = OneHot(x, vocabSize);
        double[] hNew = Tanh(Add(MatVec(Wx, xVec), MatVec(Wh, h)));
        double[] logits = MatVec(Wy, hNew);
        double[] yVec = Softmax(logits);

        // Loss (cross-entropy)
        double loss = -Math.Log(yVec[target] + 1e-9);

        // Backprop
        double[] dy = (double[])yVec.Clone();
        dy[target] -= 1.0;

        double[,] dWy = Outer(dy, hNew);
        double[] dh = MatTVec(Wy, dy);

        // through tanh
        double[] dhRaw = Mul(dh, DTanh(hNew));
        double[,] dWx = Outer(dhRaw, xVec);
        double[,] dWh = Outer(dhRaw, h);

        // SGD update
        Wy = Sub(Wy, Scale(dWy, lr));
        Wx = Sub(Wx, Scale(dWx, lr));
        Wh = Sub(Wh, Scale(dWh, lr));

        h = hNew; // carry state
        return loss;
    }

    public int Sample(int x, double temperature = 1.0, int topK = 0)
    {
        double[] xVec = OneHot(x, vocabSize);
        h = Tanh(Add(MatVec(Wx, xVec), MatVec(Wh, h)));
        double[] logits = MatVec(Wy, h);

        // Temperature + top-k sampling for variety
        ApplyTemperatureInPlace(logits, temperature);
        if (topK > 0) KeepTopKInPlace(logits, topK);

        double[] probs = Softmax(logits);
        return SampleFromDistribution(probs);
    }

    // --- helper math ---
    private static double[,] RandomMatrix(int rows, int cols, int seed)
    {
        var r = new Random(seed);
        var m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = (r.NextDouble() - 0.5) * 0.1;
        return m;
    }

    private static double[] OneHot(int idx, int size)
    {
        var v = new double[size];
        if (idx >= 0 && idx < size) v[idx] = 1.0;
        return v;
    }

    private static double[] MatVec(double[,] M, double[] v)
    {
        int rows = M.GetLength(0), cols = M.GetLength(1);
        var r = new double[rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[i] += M[i, j] * v[j];
        return r;
    }

    private static double[] MatTVec(double[,] M, double[] v)
    {
        int rows = M.GetLength(0), cols = M.GetLength(1);
        var r = new double[cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[j] += M[i, j] * v[i];
        return r;
    }

    private static double[] Add(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] + b[i];
        return r;
    }

    private static double[] Mul(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] * b[i];
        return r;
    }

    private static double[] DTanh(double[] x)
    {
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = 1 - x[i] * x[i];
        return r;
    }

    private static double[] Tanh(double[] x)
    {
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = Math.Tanh(x[i]);
        return r;
    }

    private static double[] Softmax(double[] x)
    {
        double max = x.Max();
        var exp = x.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    private static double[,] Outer(double[] a, double[] b)
    {
        var m = new double[a.Length, b.Length];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                m[i, j] = a[i] * b[j];
        return m;
    }

    private static double[,] Scale(double[,] M, double s)
    {
        int r = M.GetLength(0), c = M.GetLength(1);
        var m2 = new double[r, c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                m2[i, j] = M[i, j] * s;
        return m2;
    }

    private static double[,] Sub(double[,] A, double[,] B)
    {
        int r = A.GetLength(0), c = A.GetLength(1);
        var m2 = new double[r, c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                m2[i, j] = A[i, j] - B[i, j];
        return m2;
    }

    private void ApplyTemperatureInPlace(double[] logits, double temperature)
    {
        if (temperature <= 0) temperature = 1.0;
        for (int i = 0; i < logits.Length; i++) logits[i] /= temperature;
    }

    private void KeepTopKInPlace(double[] logits, int k)
    {
        if (k <= 0 || k >= logits.Length) return;
        int[] idx = Enumerable.Range(0, logits.Length)
                              .OrderByDescending(i => logits[i])
                              .ToArray();
        double cutoff = logits[idx[k - 1]];
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] < cutoff) logits[i] = double.NegativeInfinity;
    }

    private int SampleFromDistribution(double[] probs)
    {
        double s = rand.NextDouble(), cum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (s < cum) return i;
        }
        return probs.Length - 1;
    }
}

// -------------------- Utils --------------------

static class ShuffleExt
{
    public static void ShuffleInPlace<T>(this IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
