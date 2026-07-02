document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const steps = Array.from(document.querySelectorAll(".form-step"));
    const circles = Array.from(document.querySelectorAll(".step-circle"));
    const progress = document.getElementById("progress");
    
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const submitBtn = document.getElementById("submitBtn");
    
    const loader = document.getElementById("loader");
    const resultCard = document.getElementById("resultCard");
    
    let currentStep = 1;

    // Wizard Controls 
    function updateWizard() {
        steps.forEach(step => {
            step.classList.toggle("active", parseInt(step.dataset.step) === currentStep);
        });

        circles.forEach((circle, idx) => {
            circle.classList.toggle("active", idx < currentStep);
        });

        const progressPercent = ((currentStep - 1) / (circles.length - 1)) * 100;
        progress.style.width = `${progressPercent}%`;

        // Handle navigation buttons toggle states
        if (currentStep === 1) {
            prevBtn.disabled = true;
            nextBtn.style.display = "block";
            submitBtn.style.display = "none";
        } else if (currentStep === steps.length) {
            prevBtn.disabled = false;
            nextBtn.style.display = "none";
            submitBtn.style.display = "block";
        } else {
            prevBtn.disabled = false;
            nextBtn.style.display = "block";
            submitBtn.style.display = "none";
        }
    }

    function validateCurrentStep() {
        const activeFields = steps[currentStep - 1].querySelectorAll("input[required]");
        let isValid = true;
        activeFields.forEach(field => {
            if (!field.checkValidity()) {
                field.reportValidity();
                isValid = false;
            }
        });
        return isValid;
    }

    nextBtn.addEventListener("click", () => {
        if (validateCurrentStep() && currentStep < steps.length) {
            currentStep++;
            updateWizard();
        }
    });

    prevBtn.addEventListener("click", () => {
        if (currentStep > 1) {
            currentStep--;
            updateWizard();
        }
    });

    // Form submission processing and One-Hot Encoding map mapping
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!validateCurrentStep()) return;

        form.style.display = "none";
        loader.style.display = "block";

        const formData = new FormData(form);
        const rawData = Object.fromEntries(formData.entries());

        // Structure the precise structural base matching your FastAPI Student Model
        const payload = {
            age: parseInt(rawData.age),
            gender: parseInt(rawData.gender),
            cgpa: parseFloat(rawData.cgpa),
            internships_count: parseInt(rawData.internships_count),
            projects_count: parseInt(rawData.projects_count),
            certifications_count: parseInt(rawData.certifications_count),
            coding_skill_score: parseFloat(rawData.coding_skill_score),
            aptitude_score: parseFloat(rawData.aptitude_score),
            communication_skill_score: parseFloat(rawData.communication_skill_score),
            logical_reasoning_score: parseFloat(rawData.logical_reasoning_score),
            hackathons_participated: parseInt(rawData.hackathons_participated),
            github_repos: parseInt(rawData.github_repos),
            linkedin_connections: parseInt(rawData.linkedin_connections),
            mock_interview_score: parseFloat(rawData.mock_interview_score),
            attendance_percentage: parseFloat(rawData.attendance_percentage),
            backlogs: parseInt(rawData.backlogs),
            extracurricular_score: parseFloat(rawData.extracurricular_score),
            leadership_score: parseFloat(rawData.leadership_score),
            volunteer_experience: parseInt(rawData.volunteer_experience),
            sleep_hours: parseFloat(rawData.sleep_hours),
            study_hours_per_day: parseFloat(rawData.study_hours_per_day),
            
            // Set all One-Hot fields default down to 0
            branch_CSE: 0, branch_Civil: 0, branch_ECE: 0, branch_EEE: 0, branch_IT: 0, branch_Mechanical: 0,
            college_tier_Tier_1: 0, college_tier_Tier_2: 0, college_tier_Tier_3: 0
        };

        // Activate matching one-hot structural flag paths
        payload[`branch_${rawData.branch}`] = 1;
        payload[`college_tier_${rawData.college_tier}`] = 1;

        try {
            // Update URL string placeholder if your backend endpoint port differs
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("API server responded with error status");
            
            const data = await response.json();
            renderResult(data);

        } catch (error) {
            alert(`An error occurred: ${error.message}. Ensure your FastAPI instance is online.`);
            form.style.display = "block";
        } finally {
            loader.style.display = "none";
        }
    });

    function renderResult(data) {
        const statusEl = document.getElementById("resultStatus");
        const barEl = document.getElementById("confidenceBar");
        const textEl = document.getElementById("resultConfidence");

        statusEl.textContent = data.placement_status;
        statusEl.className = "result-status " + (data.prediction === 1 ? "placed" : "not-placed");
        
        const conf = data.confidence !== null ? data.confidence : 100;
        barEl.style.width = `${conf}%`;
        
        // Dynamically color change status visual accent lines based on confidence health levels
        barEl.style.backgroundColor = data.prediction === 1 ? "var(--success)" : "var(--danger)";
        textEl.textContent = data.confidence !== null ? `Confidence metrics assessment score: ${conf}%` : "Confidence metric unavailable";

        resultCard.style.display = "block";
    }

    document.getElementById("resetBtn").addEventListener("click", () => {
        resultCard.style.display = "none";
        form.reset();
        currentStep = 1;
        updateWizard();
        form.style.display = "block";
    });
});