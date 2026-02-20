/**
 * Workflow input validator (catalog-backed, no bundler required).
 *
 * This adapter keeps the backend workflow catalog as source of truth:
 * - Loads per-task JSON Schema from `/api/workflow/catalog` via WorkflowCatalog.
 * - Performs lightweight client-side validation before submit.
 * - Throws clear errors for actionable frontend feedback.
 */
(function () {
    function isPlainObject(value) {
        return typeof value === "object" && value !== null && !Array.isArray(value);
    }

    function compareNumber(value, limit, operator) {
        if (!Number.isFinite(limit)) {
            return true;
        }
        if (operator === "min") {
            return value >= limit;
        }
        if (operator === "max") {
            return value <= limit;
        }
        if (operator === "exclusiveMin") {
            return value > limit;
        }
        if (operator === "exclusiveMax") {
            return value < limit;
        }
        return true;
    }

    class WorkflowInputValidator {
        static async getTaskSchema(apiBase, taskType) {
            if (!window.WorkflowCatalog?.load) {
                return null;
            }
            const catalog = await window.WorkflowCatalog.load(apiBase);
            return catalog?.tasks?.[taskType]?.input_schema ?? null;
        }

        static async validateTaskInputs(apiBase, taskType, inputs) {
            const schema = await WorkflowInputValidator.getTaskSchema(apiBase, taskType);
            if (!schema) {
                // Catalog unavailable or task not present: do not block submit.
                return { valid: true, errors: [], skipped: true };
            }

            const errors = [];
            WorkflowInputValidator.validateSchema(schema, inputs, "inputs", schema, errors);
            return { valid: errors.length === 0, errors, skipped: false };
        }

        static formatErrors(errors, maxErrors = 6) {
            if (!Array.isArray(errors) || errors.length === 0) {
                return "";
            }
            const lines = errors.slice(0, maxErrors).map((entry) => {
                const path = entry?.path || "inputs";
                const message = entry?.message || "Invalid value.";
                return `${path}: ${message}`;
            });
            if (errors.length > maxErrors) {
                lines.push(`...and ${errors.length - maxErrors} more issue(s).`);
            }
            return lines.join("\n");
        }

        static buildValidationErrorMessage(taskType, errors) {
            const details = WorkflowInputValidator.formatErrors(errors);
            return details
                ? `Input validation failed for ${taskType}:\n${details}`
                : `Input validation failed for ${taskType}.`;
        }

        static async assertTaskInputs(apiBase, taskType, inputs) {
            const validation = await WorkflowInputValidator.validateTaskInputs(apiBase, taskType, inputs);
            if (validation.valid) {
                return;
            }
            throw new Error(
                WorkflowInputValidator.buildValidationErrorMessage(taskType, validation.errors)
            );
        }

        static resolveRef(rootSchema, refValue) {
            if (typeof refValue !== "string" || !refValue.startsWith("#/")) {
                return null;
            }
            const parts = refValue.slice(2).split("/");
            let current = rootSchema;
            for (const rawPart of parts) {
                const part = rawPart.replace(/~1/g, "/").replace(/~0/g, "~");
                if (!isPlainObject(current) || !(part in current)) {
                    return null;
                }
                current = current[part];
            }
            return current;
        }

        static validateSchema(schema, value, path, rootSchema, errors) {
            if (!isPlainObject(schema)) {
                return;
            }

            if (schema.$ref) {
                const resolved = WorkflowInputValidator.resolveRef(rootSchema, schema.$ref);
                if (!resolved) {
                    errors.push({
                        path,
                        message: `Unable to resolve schema reference ${schema.$ref}.`,
                    });
                    return;
                }
                WorkflowInputValidator.validateSchema(resolved, value, path, rootSchema, errors);
                return;
            }

            if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
                for (const option of schema.anyOf) {
                    const branchErrors = [];
                    WorkflowInputValidator.validateSchema(
                        option,
                        value,
                        path,
                        rootSchema,
                        branchErrors
                    );
                    if (branchErrors.length === 0) {
                        return;
                    }
                }
                errors.push({
                    path,
                    message: "Value does not match any allowed schema variant.",
                });
                return;
            }

            if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
                let validCount = 0;
                for (const option of schema.oneOf) {
                    const branchErrors = [];
                    WorkflowInputValidator.validateSchema(
                        option,
                        value,
                        path,
                        rootSchema,
                        branchErrors
                    );
                    if (branchErrors.length === 0) {
                        validCount += 1;
                    }
                }
                if (validCount !== 1) {
                    errors.push({
                        path,
                        message: "Value must match exactly one schema variant.",
                    });
                }
                return;
            }

            if (schema.const !== undefined && value !== schema.const) {
                errors.push({
                    path,
                    message: `Expected value ${JSON.stringify(schema.const)}.`,
                });
                return;
            }

            if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
                errors.push({
                    path,
                    message: `Expected one of: ${schema.enum.map((item) => JSON.stringify(item)).join(", ")}.`,
                });
                return;
            }

            if (Array.isArray(schema.type)) {
                const matchesAny = schema.type.some((candidateType) => {
                    const branchErrors = [];
                    WorkflowInputValidator.validateSchema(
                        { ...schema, type: candidateType },
                        value,
                        path,
                        rootSchema,
                        branchErrors
                    );
                    return branchErrors.length === 0;
                });
                if (!matchesAny) {
                    errors.push({
                        path,
                        message: `Expected one of types: ${schema.type.join(", ")}.`,
                    });
                }
                return;
            }

            const schemaType = schema.type;
            if (!schemaType) {
                return;
            }

            if (schemaType === "object") {
                WorkflowInputValidator.validateObjectSchema(schema, value, path, rootSchema, errors);
                return;
            }
            if (schemaType === "array") {
                WorkflowInputValidator.validateArraySchema(schema, value, path, rootSchema, errors);
                return;
            }
            if (schemaType === "string") {
                WorkflowInputValidator.validateStringSchema(schema, value, path, errors);
                return;
            }
            if (schemaType === "integer" || schemaType === "number") {
                WorkflowInputValidator.validateNumberSchema(
                    schema,
                    value,
                    path,
                    errors,
                    schemaType === "integer"
                );
                return;
            }
            if (schemaType === "boolean") {
                if (typeof value !== "boolean") {
                    errors.push({ path, message: "Expected a boolean value." });
                }
                return;
            }
            if (schemaType === "null") {
                if (value !== null) {
                    errors.push({ path, message: "Expected null." });
                }
            }
        }

        static validateObjectSchema(schema, value, path, rootSchema, errors) {
            if (!isPlainObject(value)) {
                errors.push({ path, message: "Expected an object." });
                return;
            }

            const properties = isPlainObject(schema.properties) ? schema.properties : {};
            const requiredFields = Array.isArray(schema.required) ? schema.required : [];

            for (const field of requiredFields) {
                if (value[field] === undefined) {
                    errors.push({
                        path: `${path}.${field}`,
                        message: "Field is required.",
                    });
                }
            }

            for (const [field, fieldSchema] of Object.entries(properties)) {
                if (value[field] !== undefined) {
                    WorkflowInputValidator.validateSchema(
                        fieldSchema,
                        value[field],
                        `${path}.${field}`,
                        rootSchema,
                        errors
                    );
                }
            }

            const additionalProperties = schema.additionalProperties;
            if (additionalProperties === false) {
                for (const field of Object.keys(value)) {
                    if (!(field in properties)) {
                        errors.push({
                            path: `${path}.${field}`,
                            message: "Unknown field is not allowed.",
                        });
                    }
                }
            } else if (isPlainObject(additionalProperties)) {
                for (const [field, fieldValue] of Object.entries(value)) {
                    if (!(field in properties)) {
                        WorkflowInputValidator.validateSchema(
                            additionalProperties,
                            fieldValue,
                            `${path}.${field}`,
                            rootSchema,
                            errors
                        );
                    }
                }
            }
        }

        static validateArraySchema(schema, value, path, rootSchema, errors) {
            if (!Array.isArray(value)) {
                errors.push({ path, message: "Expected an array." });
                return;
            }

            if (Number.isFinite(schema.minItems) && value.length < schema.minItems) {
                errors.push({
                    path,
                    message: `Array must contain at least ${schema.minItems} item(s).`,
                });
            }
            if (Number.isFinite(schema.maxItems) && value.length > schema.maxItems) {
                errors.push({
                    path,
                    message: `Array must contain at most ${schema.maxItems} item(s).`,
                });
            }

            if (schema.items) {
                value.forEach((item, index) => {
                    WorkflowInputValidator.validateSchema(
                        schema.items,
                        item,
                        `${path}[${index}]`,
                        rootSchema,
                        errors
                    );
                });
            }
        }

        static validateStringSchema(schema, value, path, errors) {
            if (typeof value !== "string") {
                errors.push({ path, message: "Expected a string." });
                return;
            }

            if (Number.isFinite(schema.minLength) && value.length < schema.minLength) {
                errors.push({
                    path,
                    message: `String must be at least ${schema.minLength} character(s).`,
                });
            }
            if (Number.isFinite(schema.maxLength) && value.length > schema.maxLength) {
                errors.push({
                    path,
                    message: `String must be at most ${schema.maxLength} character(s).`,
                });
            }
            if (typeof schema.pattern === "string") {
                try {
                    const pattern = new RegExp(schema.pattern);
                    if (!pattern.test(value)) {
                        errors.push({
                            path,
                            message: "String does not match required pattern.",
                        });
                    }
                } catch (_error) {
                    // Ignore invalid patterns from schema hints to avoid blocking user flows.
                }
            }
        }

        static validateNumberSchema(schema, value, path, errors, integerOnly) {
            if (typeof value !== "number" || !Number.isFinite(value)) {
                errors.push({
                    path,
                    message: integerOnly ? "Expected an integer." : "Expected a number.",
                });
                return;
            }
            if (integerOnly && !Number.isInteger(value)) {
                errors.push({ path, message: "Expected an integer." });
                return;
            }

            if (!compareNumber(value, schema.minimum, "min")) {
                errors.push({ path, message: `Must be >= ${schema.minimum}.` });
            }
            if (!compareNumber(value, schema.maximum, "max")) {
                errors.push({ path, message: `Must be <= ${schema.maximum}.` });
            }
            if (!compareNumber(value, schema.exclusiveMinimum, "exclusiveMin")) {
                errors.push({ path, message: `Must be > ${schema.exclusiveMinimum}.` });
            }
            if (!compareNumber(value, schema.exclusiveMaximum, "exclusiveMax")) {
                errors.push({ path, message: `Must be < ${schema.exclusiveMaximum}.` });
            }
        }
    }

    window.WorkflowInputValidator = WorkflowInputValidator;
})();
