if (window.tailwind) {
    window.tailwind.config = {
        darkMode: 'class',
        theme: {
            extend: {
                colors: {
                    background: '#09090b',
                    foreground: '#fafafa',
                    card: {
                        DEFAULT: 'rgba(24, 24, 27, 0.4)',
                        foreground: '#fafafa',
                    },
                    muted: {
                        DEFAULT: '#27272a',
                        foreground: '#a1a1aa',
                    },
                    accent: {
                        DEFAULT: '#6366f1',
                        foreground: '#ffffff',
                    },
                    border: '#27272a',
                },
                fontFamily: {
                    sans: ['Inter', 'sans-serif'],
                }
            }
        }
    }
} else {
    console.warn("Tailwind CSS not loaded.");
}

// --- DATA ARCHITECTURE ---
const curriculumData = [
    // Year 1 Sem 1
    {
        year: "1st Year", semester: "1st Semester", name: "Linear Algebra", category: "Algebra", bibliography: [
            "Lay, D. C., Linear Algebra and its Applications, 4th Ed., Pearson, 2012.",
            "Axler, S., Linear Algebra Done Right, Springer.",
            "Strang, G., Linear Algebra and Its Applications, Academic Press."
        ]
    },
    {
        year: "1st Year", semester: "1st Semester", name: "Single-Variate Calculus", category: "Analysis", bibliography: [
            "Apostol, T. M., Calculus, Vol. I, Reverté, 1994.",
            "Stewart, J., Calculus, Cengage Learning.",
            "Spivak, M., Calculus, Cambridge University Press."
        ]
    },
    {
        year: "1st Year", semester: "1st Semester", name: "Combinatorics and Graphs", category: "Statistics", bibliography: [
            "Bona, M., A Walk Through Combinatorics: An Introduction to Enumeration and Graph Theory, World Scientific.",
            "Brualdi, R. A., Introductory Combinatorics, Pearson.",
            "Bondy, J. A., & Murty, U. S. R., Graph Theory, Springer."
        ], readings: ["Quick check page 26", "Quick check page 33", "Exercises chapter 1"]
    },
    {
        year: "1st Year", semester: "1st Semester", name: "Introduction to Number Theory", category: "Algebra", bibliography: [
            "Hardy, G. H., & Wright, E. M., An Introduction to the Theory of Numbers, Oxford University Press.",
            "Tattersall, J. J., Elementary Number Theory, Cambridge University Press.",
            "Rosen, K., Elementary Number Theory and Its Applications, Pearson."
        ]
    },
    {
        year: "1st Year", semester: "1st Semester", name: "Experimental Mathematics", category: "Computation", bibliography: [
            "Borwein, J., & Bailey, D., Mathematics by Experiment, A K Peters.",
            "Wolfram, S., The Mathematica Book, Cambridge University Press.",
            "Blanchman, N., Mathematica: a Practical Approach, Prentice Hall."
        ]
    },

    // Year 1 Sem 2
    {
        year: "1st Year", semester: "2nd Semester", name: "Introduction to Abstract Algebra", category: "Algebra", bibliography: [
            "Lang, S., Undergraduate Algebra, Springer.",
            "Herstein, I. N., Topics in Algebra, Wiley.",
            "Fraleigh, J. B., A First Course in Abstract Algebra, Pearson."
        ]
    },
    {
        year: "1st Year", semester: "2nd Semester", name: "Multi-Variate Calculus", category: "Analysis", bibliography: [
            "Apostol, T. M., Calculus, Vol. II, Reverté.",
            "Marsden, J. E., & Tromba, A. J., Vector Calculus, Freeman.",
            "Stewart, J., Multivariable Calculus, Cengage Learning."
        ]
    },
    {
        year: "1st Year", semester: "2nd Semester", name: "Numerical Optimization", category: "Computation", bibliography: [
            "Luenberger, D. G., & Ye, Y., Linear and Nonlinear Programming, Springer.",
            "Matousek, J., & Gartner, B., Understanding and Using Linear Programming, Springer.",
            "Nocedal, J. & Wright, S. J., Numerical Optimization, Springer."
        ]
    },
    {
        year: "1st Year", semester: "2nd Semester", name: "Real Analysis", category: "Analysis", bibliography: [
            "Benedetto, J. J., Real Variable and Integration, Teubner, 1976.",
            "Bartle, R. G., The Elements of Real Analysis, Wiley.",
            "Rudin, W., Principles of Mathematical Analysis, McGraw-Hill."
        ]
    },
    {
        year: "1st Year", semester: "2nd Semester", name: "Introduction to Geometry", category: "Algebra", bibliography: [
            "Audin, M., Geometry, Springer, 2002.",
            "Reid, M., & Szendroi, B., Geometry and Topology, Cambridge University Press.",
            "Pressley, A., Elementary Differential Geometry, Springer."
        ]
    },

    // Year 2 Sem 1
    {
        year: "2nd Year", semester: "1st Semester", name: "Vector and Manifold Calculus", category: "Analysis", bibliography: [
            "Spivak, M., Calculus on Manifolds, Benjamin/Cummings.",
            "Munkres, J. R., Analysis on Manifolds, Addison-Wesley.",
            "Fleming, W., Functions of Several Variables, Springer."
        ]
    },
    {
        year: "2nd Year", semester: "1st Semester", name: "Introduction to Complex Analysis", category: "Analysis", bibliography: [
            "Marsden, J. E., & Hoffman, M. J., Basic Complex Analysis, Freeman.",
            "Ahlfors, L. V., Complex Analysis, McGraw-Hill.",
            "Stein, E. M., & Shakarchi, R., Complex Analysis, Princeton University Press."
        ]
    },
    {
        year: "2nd Year", semester: "1st Semester", name: "Probability and Statistics", category: "Statistics", bibliography: [
            "Ross, S. M., Introduction to Probability and Statistics, Academic Press.",
            "Montgomery, D. C., Applied Statistics and Probability, Wiley.",
            "Hogg, R. V., & Tanis, E. A., Probability and Statistical Inference, Pearson."
        ]
    },
    {
        year: "2nd Year", semester: "1st Semester", name: "Numeric Linear Algebra", category: "Computation", bibliography: [
            "Golub, G. H., Matrix Computations, Johns Hopkins University Press.",
            "Trefethen, L. N., & Bau, D., Numerical Linear Algebra, SIAM.",
            "Demmel, J. W., Applied Numerical Linear Algebra, SIAM."
        ]
    },
    {
        year: "2nd Year", semester: "1st Semester", name: "Linear Analysis", category: "Analysis", bibliography: [
            "Kreyszig, E., Advanced Engineering Mathematics, Wiley.",
            "Pinsky, M. A., Fourier Analysis and Wavelets, Brooks/Cole.",
            "Stein, E. M., & Shakarchi, R., Fourier Analysis, Princeton University Press."
        ]
    },

    // Year 2 Sem 2
    {
        year: "2nd Year", semester: "2nd Semester", name: "Abstract Algebra: Groups, Rings, Fields", category: "Algebra", bibliography: [
            "Lang, S., Algebra, Springer.",
            "Rotman, J. J., Theory of Groups, Springer.",
            "Dummit, D. S., & Foote, R. M., Abstract Algebra, Wiley."
        ]
    },
    {
        year: "2nd Year", semester: "2nd Semester", name: "Advanced Probability and Statistics", category: "Statistics", bibliography: [
            "Casella, G., & Berger, R. L., Statistical Inference, Duxbury.",
            "DeGroot, M. H., & Schervish, M. J., Probability and Statistics, Addison-Wesley.",
            "Wasserman, L., All of Statistics, Springer."
        ]
    },
    {
        year: "2nd Year", semester: "2nd Semester", name: "Functional Linear Analysis", category: "Analysis", bibliography: [
            "Kreyszig, E., Introductory Functional Analysis, Wiley.",
            "Conway, J. B., A Course in Functional Analysis, Springer.",
            "Brezis, H., Functional Analysis, Springer."
        ]
    },
    {
        year: "2nd Year", semester: "2nd Semester", name: "Computational Mathematics", category: "Computation", bibliography: [
            "Burden, R. L., & Faires, J. D., Numerical Analysis, Cengage Learning.",
            "Heath, M. T., Scientific Computing, McGraw-Hill.",
            "Quarteroni, A., et al., Numerical Mathematics, Springer."
        ]
    },
    {
        year: "2nd Year", semester: "2nd Semester", name: "Algorithms and Comp. Modelling", category: "Computation", bibliography: [
            "Cormen, T. H., et al., Introduction to Algorithms, MIT Press.",
            "Kleinberg, J. & Tardos, E., Algorithm Design, Pearson."
        ]
    },

    // Year 3 Sem 1
    {
        year: "3rd Year", semester: "1st Semester", name: "Intro to Computational Complexity", category: "Computation", bibliography: [
            "Arora, S., & Barak, B., Computational Complexity, Cambridge Univ. Press.",
            "Sipser, M., Theory of Computation, Cengage Learning.",
            "Cormen, T. H., et al., Introduction to Algorithms, MIT Press."
        ]
    },
    {
        year: "3rd Year", semester: "1st Semester", name: "Linear Model Analysis", category: "Statistics", bibliography: [
            "Kutner, M., et al., Applied Linear Statistical Models, McGraw-Hill.",
            "Faraway, J. J., Linear Models with R, CRC Press.",
            "Montgomery, D. C., et al., Linear Regression Analysis, Wiley."
        ]
    },
    {
        year: "3rd Year", semester: "1st Semester", name: "Multivariate Analysis", category: "Statistics", bibliography: [
            "Johnson, R. A., & Wichern, D. W., Multivariate Analysis, Pearson.",
            "Mardía, K. V., et al., Multivariate Analysis, Academic Press.",
            "Schabenberger, O., Statistical Methods for Spatial Data, CRC/Chapman."
        ]
    },
    {
        year: "3rd Year", semester: "1st Semester", name: "Numerical Analysis", category: "Computation", bibliography: [
            "Kress, R., Numerical Analysis, Springer.",
            "Quarteroni, A., et al., Numerical Mathematics, Springer.",
            "Atkinson, K., Theoretical Numerical Analysis, Springer."
        ]
    },
    {
        year: "3rd Year", semester: "1st Semester", name: "Probability Theory", category: "Statistics", bibliography: [
            "Karr, A. F., Probability, Springer.",
            "Resnick, S. I., A Probability Path, Birkhäuser.",
            "Billingsley, P., Probability and Measure, Wiley."
        ]
    },
    {
        year: "3rd Year", semester: "1st Semester", name: "Topology", category: "Algebra", bibliography: [
            "Munkres, J. R., Topology, Prentice Hall.",
            "Dugundji, J., Topology, Allyn and Bacon.",
            "Kelley, J. L., General Topology, Springer."
        ]
    },

    // Year 3 Sem 2
    {
        year: "3rd Year", semester: "2nd Semester", name: "Stochastic Processes", category: "Statistics", bibliography: [
            "Ross, S. M., Stochastic Processes, Wiley.",
            "Kulkarni, V. G., Analysis of Stochastic Systems, Chapman & Hall.",
            "Karlin, S. & Taylor, H., Stochastic Processes, Academic Press."
        ]
    },
    {
        year: "3rd Year", semester: "2nd Semester", name: "Measure and Integration", category: "Analysis", bibliography: [
            "Folland, G. B., Real Analysis, Wiley.",
            "Rudin, W., Real and Complex Analysis, McGraw-Hill.",
            "Royden, H. L., Real Analysis, Macmillan."
        ]
    },
    {
        year: "3rd Year", semester: "2nd Semester", name: "Reliability and Quality Control", category: "Statistics", bibliography: [
            "Montgomery, D. C., Statistical Quality Control, Wiley.",
            "Barlow, R. E., Mathematical Theory of Reliability, SIAM."
        ]
    },
    {
        year: "3rd Year", semester: "2nd Semester", name: "Mathematical Finance", category: "Statistics", bibliography: [
            "Wilmott, P., Quantitative Finance, Wiley.",
            "Hull, J. C., Options, Futures, and Other Derivatives, Pearson.",
            "Mikosch, T., Elementary Stochastic Calculus, World Scientific."
        ]
    },
    {
        year: "3rd Year", semester: "2nd Semester", name: "Statistical Data Mining", category: "Statistics", bibliography: [
            "Tan, P.-N., et al., Introduction to Data Mining, Pearson.",
            "Hastie, T., et al., The Elements of Statistical Learning, Springer.",
            "Hand, D., et al., Principles of Data Mining, MIT Press."
        ]
    },
    {
        year: "3rd Year", semester: "2nd Semester", name: "Statistics and Data Science", category: "Statistics", bibliography: [
            "James, G., et al., Statistical Learning, Springer.",
            "Matloff, N., Probability and Statistics for Data Science, CRC Press.",
            "Bruce, P. & Bruce, A., Practical Statistics for Data Scientists, O'Reilly."
        ]
    }
];

// --- GEMINI API INTEGRATION ---
const apiKey = "";

async function callGemini(prompt) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
    const payload = {
        contents: [{ parts: [{ text: prompt }] }]
    };

    let delay = 1000;
    for (let i = 0; i < 5; i++) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();
            return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";
        } catch (error) {
            if (i === 4) throw error;
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= 2;
        }
    }
}

async function getCourseBrief(courseName) {
    showModal("✨ Generating Intelligent Brief...", `<div class="flex justify-center py-8"><div class="spinner"></div></div>`);
    try {
        const prompt = `You are an expert math professor. Explain what the course "${courseName}" is about, why it is fundamental to a mathematics degree, and give one distinct real-world application (e.g. in tech, finance, or physics). Be concise (under 120 words). Format with bold headers.`;
        const result = await callGemini(prompt);
        updateModal(courseName + " Smart Brief", result);
    } catch (err) {
        updateModal("Error", "Failed to connect to Gemini API. Please check your network or API key.");
    }
}

async function generateCareerStrategy() {
    const output = document.getElementById('careerOutput');
    const btn = document.getElementById('careerBtn');
    output.classList.remove('hidden');
    output.innerHTML = `<div class="flex items-center gap-3"><div class="spinner"></div><span>AI is analyzing the curriculum...</span></div>`;
    btn.disabled = true;
    btn.classList.add('opacity-50');

    try {
        const courseList = curriculumData.map(c => c.name).join(', ');
        const prompt = `Based on this mathematics curriculum: [${courseList}], suggest 3 specific modern career paths or research specializations (e.g. Quantitative Finance, AI Research, Cryptography). For each, explain briefly why it matches and which core courses are the most critical. Use a professional, inspiring tone. Keep it concise.`;
        const result = await callGemini(prompt);
        output.innerHTML = `<div class="whitespace-pre-wrap">${result}</div>`;
    } catch (err) {
        output.innerHTML = "Error generating roadmap. Please try again later.";
    } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-50');
    }
}

// --- APP UI LOGIC ---

function navigate(section) {
    document.getElementById('section-dashboard').classList.add('hidden');
    document.getElementById('section-explorer').classList.add('hidden');
    document.getElementById('nav-dashboard').classList.remove('nav-item-active');
    document.getElementById('nav-explorer').classList.remove('nav-item-active');

    document.getElementById('section-' + section).classList.remove('hidden');
    document.getElementById('nav-' + section).classList.add('nav-item-active');

    if (section === 'dashboard') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function showModal(title, content) {
    document.getElementById('modalTitle').innerText = title;
    document.getElementById('modalContent').innerHTML = content;
    document.getElementById('aiModal').classList.remove('hidden');
    document.getElementById('aiModal').classList.add('flex');
}

function updateModal(title, content) {
    document.getElementById('modalTitle').innerText = title;
    document.getElementById('modalContent').innerHTML = `<div class="whitespace-pre-wrap">${content}</div>`;
}

function closeModal() {
    document.getElementById('aiModal').classList.add('hidden');
    document.getElementById('aiModal').classList.remove('flex');
}

// Label wrapping utility (16 chars logic)
function wrapLabel(label) {
    if (label.length <= 16) return label;
    const words = label.split(' ');
    const lines = [];
    let currentLine = words[0];

    for (let i = 1; i < words.length; i++) {
        if ((currentLine + ' ' + words[i]).length <= 18) {
            currentLine += ' ' + words[i];
        } else {
            lines.push(currentLine);
            currentLine = words[i];
        }
    }
    lines.push(currentLine);
    return lines;
}

const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        tooltip: {
            backgroundColor: '#18181b',
            titleColor: '#ffffff',
            bodyColor: '#a1a1aa',
            borderColor: '#27272a',
            borderWidth: 1,
            padding: 12,
            callbacks: {
                title: function (tooltipItems) {
                    const item = tooltipItems[0];
                    let label = item.chart.data.labels[item.dataIndex];
                    return Array.isArray(label) ? label.join(' ') : label;
                }
            }
        },
        legend: {
            display: false
        }
    },
    scales: {
        x: {
            grid: { color: '#18181b' },
            ticks: { color: '#71717a', font: { size: 10 } }
        },
        y: {
            grid: { color: '#18181b' },
            ticks: { color: '#71717a', font: { size: 10 } }
        }
    }
};

function initCharts() {
    const catCounts = curriculumData.reduce((acc, curr) => {
        acc[curr.category] = (acc[curr.category] || 0) + 1;
        return acc;
    }, {});

    new Chart(document.getElementById('subjectChart').getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: Object.keys(catCounts).map(wrapLabel),
            datasets: [{
                data: Object.values(catCounts),
                backgroundColor: ['#6366f1', '#a855f7', '#ec4899', '#06b6d4'],
                borderWidth: 0,
                hoverOffset: 10
            }]
        },
        options: {
            ...commonOptions,
            cutout: '70%',
            plugins: {
                ...commonOptions.plugins,
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: { color: '#a1a1aa', usePointStyle: true, padding: 20 }
                }
            }
        }
    });

    const semesterMap = curriculumData.reduce((acc, curr) => {
        const key = curr.year + ' ' + curr.semester;
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    new Chart(document.getElementById('loadChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: Object.keys(semesterMap).map(k => wrapLabel(k.replace('Semester', 'Sem'))),
            datasets: [{
                label: 'Courses',
                data: Object.values(semesterMap),
                backgroundColor: '#6366f1',
                borderRadius: 4
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, beginAtZero: true, ticks: { stepSize: 1 } }
            }
        }
    });
}

function renderGrid() {
    const container = document.getElementById('courseGrid');
    const noRes = document.getElementById('noResults');
    const yearVal = document.getElementById('yearFilter').value;
    const catVal = document.getElementById('categoryFilter').value;
    const searchVal = document.getElementById('searchInput').value.toLowerCase();

    const filtered = curriculumData.filter(c => {
        const matchYear = yearVal === 'All' || c.year === yearVal;
        const matchCat = catVal === 'All' || c.category === catVal;
        const matchSearch = c.name.toLowerCase().includes(searchVal);
        return matchYear && matchCat && matchSearch;
    });

    container.innerHTML = '';

    if (filtered.length === 0) {
        noRes.classList.remove('hidden');
        return;
    }
    noRes.classList.add('hidden');

    filtered.forEach(course => {
        const card = document.createElement('div');
        card.className = 'glass-card p-6 flex flex-col h-full group';

        const bibItems = course.bibliography.map(b => `<li class="mb-2 pl-3 border-l border-muted-foreground/30 text-xs text-muted-foreground">${b}</li>`).join('');

        card.innerHTML = `
                    <div class="flex justify-between items-start mb-4">
                        <span class="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">${course.year} • ${course.semester.replace('Semester', 'Sem')}</span>
                        <div class="flex gap-2">
                             <span class="px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-tighter bg-accent/10 text-accent border border-accent/20">${course.category}</span>
                        </div>
                    </div>
                    <h5 class="text-base font-bold mb-4 group-hover:text-accent transition-colors">${course.name}</h5>
                    <div class="flex-grow space-y-4">
                        <details class="group/details">
                            <summary class="list-none flex items-center gap-2 text-[10px] font-bold text-muted-foreground hover:text-foreground cursor-pointer transition-colors uppercase tracking-widest">
                                <span class="group-open/details:rotate-90 transition-transform">▸</span> Bibliography
                            </summary>
                            <ul class="mt-4 list-none">
                                ${bibItems}
                            </ul>
                        </details>
                    </div>
                    <div class="mt-6 pt-4 border-t border-border">
                        <button onclick="getCourseBrief('${course.name}')" class="w-full py-2 bg-accent/5 hover:bg-accent/10 border border-accent/20 text-[10px] font-bold uppercase tracking-widest text-accent rounded-lg transition-all flex items-center justify-center gap-2">
                            ✨ Smart Brief
                        </button>
                    </div>
                `;
        container.appendChild(card);
    });
}

// --- INIT ---
// --- INIT ---
function initApp() {
    initCharts();
    renderGrid();

    // Use optional chaining or check existence to prevent errors if elements aren't found immediately
    const yearFilter = document.getElementById('yearFilter');
    const categoryFilter = document.getElementById('categoryFilter');
    const searchInput = document.getElementById('searchInput');

    if (yearFilter) yearFilter.addEventListener('change', renderGrid);
    if (categoryFilter) categoryFilter.addEventListener('change', renderGrid);
    if (searchInput) searchInput.addEventListener('input', renderGrid);
}

// Check if DOM is already loaded (which it is for dynamically injected scripts)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
